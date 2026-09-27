"""Fine-tune ResNet50 + BiLSTM + Attention (best2.pt) on labelled webcam clips.

Spatial features: ResNet50 per frame. Temporal features: BiLSTM + attention
over 16 consecutive frames. Same architecture and checkpoint format as
train_resnet50_bilstm_attention_clean.py, so the web app loads the result.

Kaggle / Colab (GPU):
  1. Upload data/webcam_dataset (violence/, non_violence/) and best2.pt as datasets.
  2. Set WEBCAM_ROOT, INIT_CHECKPOINT and EXTRA_DATASETS (RWF-2000, RLVS, AIRTLab).
  3. Run. Copy OUT_DIR/best_webcam.pt to checkpoints/best_webcam.pt.
Local: python notebooks/finetune_webcam.py  (works on CPU, only slower).
"""
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50

# ===== 1. CONFIG =====
ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
WEBCAM_ROOT = ROOT / "data" / "webcam_dataset"          # Kaggle: /kaggle/input/<your-dataset>
INIT_CHECKPOINT = ROOT / "checkpoints" / "best2.pt"      # Kaggle: /kaggle/input/<ckpt>/best2.pt
# Mix in the datasets best2.pt was trained on so it does not forget them.
# Each entry: (folder, {subfolder: label}); videos are found recursively, so
# AIRTLab's cam1/cam2 subfolders work. Copy the exact paths from the Kaggle
# sidebar (Input -> right-click folder -> Copy path). Empty list = off.
EXTRA_DATASETS = [
    # RWF-2000 (train split only; its val split is not used here)
    # ("/kaggle/input/rwf2000/RWF-2000/train", {"Fight": 1, "NonFight": 0}),
    # Real Life Violence Situations
    # ("/kaggle/input/real-life-violence-situations-dataset/Real Life Violence Dataset",
    #  {"Violence": 1, "NonViolence": 0}),
    # AIRTLab ("DataWebCam"): violent/cam1, violent/cam2, non-violent/cam1, non-violent/cam2
    # ("/kaggle/input/datawebcam/A-Dataset-for-Automatic-Violence-Detection-in-Videos-master/violence-detection-dataset",
    #  {"violent": 1, "non-violent": 0}),
]
EXTRA_PER_EPOCH = 1.0  # extra clips drawn each epoch, as a multiple of webcam train clips
OUT_DIR = ROOT / "checkpoints" if not Path("/kaggle/working").exists() else Path("/kaggle/working")

SEED, T, SIZE = 42, 16, 224
BATCH, EPOCHS, PATIENCE = 4, 15, 4
LR_HEAD, LR_BACKBONE = 1e-4, 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".webm"}


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything()


# ===== 2. DATA =====
def split_of(path):
    """Copy of src/data/webcam_dataset.split_of — keep in sync."""
    bucket = int(hashlib.md5(Path(path).name.encode()).hexdigest(), 16) % 10
    return "test" if bucket < 2 else "val" if bucket < 3 else "train"


def webcam_clips(split):
    out = []
    for name, label in (("non_violence", 0), ("violence", 1)):
        out += [(str(p), label) for p in sorted((WEBCAM_ROOT / name).glob("*.mp4")) if split_of(p) == split]
    return out


def extra_clips():
    out = []
    for root, folders in EXTRA_DATASETS:
        for name, label in folders.items():
            folder = Path(root) / name
            if not folder.is_dir():
                raise FileNotFoundError(f"Không thấy thư mục {folder} — kiểm tra EXTRA_DATASETS")
            found = [(str(p), label) for p in sorted(folder.rglob("*")) if p.suffix.lower() in EXTS]
            print(f"  extra {folder}: {len(found)} clip, label={label}")
            out += found
    return out


# The Camera page sends every frame at <=640px as JPEG quality 0.9. The model turned out to
# be very sensitive to JPEG artefacts (a static empty room went from 0.05 to
# 0.96 "violence" after JPEG encoding), so train with random JPEG quality and
# validate at the live quality.
LIVE_JPEG_QUALITY = 90
TRAIN_JPEG_QUALITY = (50, 95)
TRAIN_JPEG_PROB = 0.8


def jpeg_roundtrip(bgr, quality):
    return cv2.imdecode(cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])[1], cv2.IMREAD_COLOR)


def read_frames(path, max_side=640, jpeg_quality=None):
    """All frames (RGB), downscaled and optionally JPEG-encoded like the webcam page sends them."""
    cap = cv2.VideoCapture(str(path)); frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        h, w = bgr.shape[:2]; s = min(1.0, max_side / max(h, w))
        if s < 1.0:
            bgr = cv2.resize(bgr, (int(w * s), int(h * s)))
        if jpeg_quality:
            bgr = jpeg_roundtrip(bgr, jpeg_quality)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"Không đọc được video: {path}")
    while len(frames) < T:
        frames.append(frames[-1])
    return frames


MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
normalize = transforms.Normalize(MEAN, STD)
# Eval = src/data/transforms.eval_transform used by the web app.
eval_frame_tf = transforms.Compose([transforms.Resize(SIZE + 32), transforms.CenterCrop(SIZE)])


def train_clip_tf(frames):
    """Same random crop/flip/colour for every frame of a clip. Wide crop scale
    covers both full webcam frames and the YOLO person crops used live."""
    imgs = [transforms.functional.resize(Image.fromarray(f), SIZE + 32) for f in frames]
    i, j, h, w = transforms.RandomResizedCrop.get_params(imgs[0], scale=(0.5, 1.0), ratio=(0.75, 1.33))
    flip = random.random() < 0.5
    b, c, s = random.uniform(0.7, 1.3), random.uniform(0.7, 1.3), random.uniform(0.8, 1.2)
    out = []
    for img in imgs:
        img = transforms.functional.resized_crop(img, i, j, h, w, (SIZE, SIZE))
        if flip:
            img = transforms.functional.hflip(img)
        img = transforms.functional.adjust_brightness(img, b)
        img = transforms.functional.adjust_contrast(img, c)
        img = transforms.functional.adjust_saturation(img, s)
        out.append(normalize(transforms.functional.to_tensor(img)))
    return torch.stack(out)


def eval_clip_tf(frames):
    return torch.stack([normalize(transforms.functional.to_tensor(eval_frame_tf(Image.fromarray(f)))) for f in frames])


class TrainDataset(Dataset):
    """Random 16 consecutive frames per clip (as in the original training)."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        quality = random.randint(*TRAIN_JPEG_QUALITY) if random.random() < TRAIN_JPEG_PROB else None
        frames = read_frames(path, jpeg_quality=quality)
        start = random.randint(0, len(frames) - T)
        return train_clip_tf(frames[start:start + T]), torch.tensor(label)


# ===== 3. MODEL (identical to the training notebook / src.models.cnn_lstm) =====
class TemporalAttention(nn.Module):
    def __init__(self, d):
        super().__init__(); self.score = nn.Sequential(nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))

    def forward(self, h):
        a = torch.softmax(self.score(h).squeeze(-1), dim=1)
        return torch.bmm(a.unsqueeze(1), h).squeeze(1), a


class ResNet50BiLSTMAttention(nn.Module):
    def __init__(self):
        super().__init__(); net = resnet50(weights=None)
        self.cnn = nn.Sequential(*list(net.children())[:-1])
        self.lstm = nn.LSTM(2048, 512, 2, batch_first=True, bidirectional=True, dropout=.5)
        self.attention = TemporalAttention(1024)
        self.head = nn.Sequential(nn.Dropout(.5), nn.Linear(1024, 2))

    def forward(self, x):
        b, t, c, h, w = x.shape; z = self.cnn(x.reshape(b * t, c, h, w)).flatten(1)
        z, _ = self.lstm(z.view(b, t, 2048)); context, alpha = self.attention(z)
        return self.head(context), alpha


def build_model():
    model = ResNet50BiLSTMAttention()
    ckpt = torch.load(INIT_CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    # Small dataset: train only the last ResNet stage + LSTM + attention + head.
    # cnn children: conv1, bn1, relu, maxpool, layer1..layer4 (index 7), avgpool.
    for idx, child in enumerate(model.cnn.children()):
        for p in child.parameters():
            p.requires_grad = idx == 7
    return model.to(DEVICE)


def set_train_mode(model):
    model.train()
    # Frozen stages keep their BatchNorm statistics.
    for idx, child in enumerate(model.cnn.children()):
        if idx != 7:
            child.eval()


# ===== 4. EVAL: clip-level, same windows as the web app =====
@torch.no_grad()
def clip_probs(model, items, stride=8):
    model.eval(); probs = []
    for path, _ in items:
        frames = read_frames(path, jpeg_quality=LIVE_JPEG_QUALITY)
        starts = list(range(0, len(frames) - T + 1, stride)) or [0]
        batch = torch.stack([eval_clip_tf(frames[s:s + T]) for s in starts]).to(DEVICE)
        p = torch.softmax(model(batch)[0], dim=1)[:, 1].mean().item()
        probs.append(p)
    return np.array(probs)


def metrics(y, probs, thr):
    pred = (probs >= thr).astype(int)
    return {"accuracy": accuracy_score(y, pred), "f1": f1_score(y, pred, average="macro", zero_division=0),
            "confusion[[TN,FP],[FN,TP]]": confusion_matrix(y, pred, labels=[0, 1]).tolist(), "threshold": thr}


def best_threshold(y, probs):
    grid = np.round(np.arange(0.3, 0.91, 0.05), 2)
    return float(max(grid, key=lambda t: (f1_score(y, (probs >= t).astype(int), average="macro", zero_division=0), -abs(t - 0.6))))


def main():
    # Windows consoles default to cp1252, which cannot print Vietnamese.
    # (Jupyter streams have no reconfigure and already handle UTF-8.)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    train_w, val, test = webcam_clips("train"), webcam_clips("val"), webcam_clips("test")
    extra = extra_clips()
    print(f"Device {DEVICE} | webcam train/val/test: {len(train_w)}/{len(val)}/{len(test)} | extra pool: {len(extra)}")
    for name, items in (("train", train_w), ("val", val), ("test", test)):
        ys = [y for _, y in items]
        print(f"  {name}: violence={sum(ys)} non_violence={len(ys) - sum(ys)}")
    if not train_w or not val or len({y for _, y in val}) < 2:
        raise SystemExit("Cần thêm dữ liệu: mỗi tập train/val phải có cả 2 lớp. Quay thêm clip rồi chạy lại.")

    model = build_model()
    y_val = np.array([y for _, y in val]); y_test = np.array([y for _, y in test])
    base_val = clip_probs(model, val)
    print("BEFORE fine-tune  val:", metrics(y_val, base_val, 0.6))

    ys = [y for _, y in train_w]
    weights = torch.tensor([len(ys) / max(1, 2 * (len(ys) - sum(ys))), len(ys) / max(1, 2 * sum(ys))], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
    optimizer = torch.optim.AdamW([
        {"params": model.cnn[7].parameters(), "lr": LR_BACKBONE},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("cnn.")], "lr": LR_HEAD},
    ], weight_decay=1e-4)

    best_f1, best_state, best_thr, stale = -1.0, None, 0.6, 0
    for epoch in range(1, EPOCHS + 1):
        items = train_w + random.sample(extra, min(len(extra), int(len(train_w) * EXTRA_PER_EPOCH)))
        loader = DataLoader(TrainDataset(items), batch_size=BATCH, shuffle=True, num_workers=2 if DEVICE.type == "cuda" else 0)
        set_train_mode(model); losses = []
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x)[0], y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            losses.append(loss.item())
        pv = clip_probs(model, val); thr = best_threshold(y_val, pv); m = metrics(y_val, pv, thr)
        print(f"epoch {epoch:02d} loss={np.mean(losses):.4f} val={m}")
        if m["f1"] > best_f1:
            best_f1, best_state, best_thr, stale = m["f1"], copy.deepcopy(model.state_dict()), thr, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    model.load_state_dict(best_state)
    out = OUT_DIR / "best_webcam.pt"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": {
        "data": {"classes": ["non_violence", "violence"], "num_frames": T, "image_size": SIZE},
        "model": {"backbone": "resnet50", "pretrained": False, "freeze_backbone": False, "lstm_hidden": 512,
                  "lstm_layers": 2, "bidirectional": True, "dropout": .5, "use_attention": True, "num_classes": 2},
        "infer": {"window_frames": T, "stride": 8, "threshold": best_thr}},
        "val": {"f1": best_f1, "threshold": best_thr}, "finetuned_on": "webcam"}, out)
    print("saved", out)
    if len(test):
        report = {"before": metrics(y_test, clip_probs(build_model(), test), 0.6),
                  "after": metrics(y_test, clip_probs(model, test), best_thr)}
        print("TEST:", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
