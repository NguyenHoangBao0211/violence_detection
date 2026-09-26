"""Train ResNet50 + BiLSTM + temporal Attention for violence detection."""

from pathlib import Path
import random
import numpy as np
import cv2
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights


# Kaggle: sửa DATA_ROOT theo dataset đã Add Input.
DATA_ROOT = Path("/kaggle/input/datasets/daniazehra/rwf-2000/RWF-2000")
OUT_DIR = Path("/kaggle/working/resnet50_bilstm_attention")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
T = 16
SIZE = 224
BATCH = 2
EPOCHS = 20
PATIENCE = 5
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".mpg", ".mpeg", ".webm"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def videos(folder):
    return sorted(
        p for p in Path(folder).rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def collect_dataset():
    nv, vv = [], []
    # RWF-2000 layout: train/Fight, train/NonFight, val/Fight, val/NonFight.
    for split in ("train", "val"):
        nv += videos(DATA_ROOT / split / "NonFight")
        vv += videos(DATA_ROOT / split / "Fight")
    if not nv or not vv:
        # Real Life Violence Dataset layout fallback.
        nv = videos(DATA_ROOT / "NonViolence")
        vv = videos(DATA_ROOT / "Violence")
    if not nv or not vv:
        raise FileNotFoundError(f"Không tìm thấy Fight/NonFight tại {DATA_ROOT}")
    return nv, vv


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if cap.isOpened() and n > 0:
        # Lấy 16 frame liên tiếp ở vị trí ngẫu nhiên để khớp webcam sliding-window.
        start = random.randint(0, max(0, n - T))
        for idx in range(start, min(start + T, n)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if ok:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    while len(frames) < T:
        frames.append(frames[-1] if frames else np.zeros((SIZE, SIZE, 3), np.uint8))
    return frames[:T]


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
TFM = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class VideoDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths, self.labels = paths, labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        clip = torch.stack([TFM(Image.fromarray(f)) for f in read_frames(self.paths[i])])
        return clip, torch.tensor(self.labels[i], dtype=torch.long)


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h):
        alpha = torch.softmax(self.score(h).squeeze(-1), dim=1)
        context = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)
        return context, alpha


class ResNet50BiLSTMAttention(nn.Module):
    def __init__(self):
        super().__init__()
        net = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(net.children())[:-1])
        self.lstm = nn.LSTM(
            2048, 512, num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.5,
        )
        self.attention = TemporalAttention(1024)
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(1024, 2))

    def forward(self, x):
        b, t, c, h, w = x.shape
        z = self.cnn(x.reshape(b * t, c, h, w)).flatten(1)
        z, _ = self.lstm(z.view(b, t, 2048))
        context, alpha = self.attention(z)
        return self.head(context), alpha


def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    losses, y_true, y_pred = [], [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            y_true.extend(y.cpu().tolist())
            y_pred.extend(logits.argmax(1).detach().cpu().tolist())
    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


nv, vv = collect_dataset()
paths = [str(p) for p in nv + vv]
labels = [0] * len(nv) + [1] * len(vv)
train_p, temp_p, train_y, temp_y = train_test_split(
    paths, labels, test_size=.30, random_state=SEED, stratify=labels
)
val_p, test_p, val_y, test_y = train_test_split(
    temp_p, temp_y, test_size=.50, random_state=SEED, stratify=temp_y
)
loader = lambda p, y, train=False: DataLoader(
    VideoDataset(p, y), batch_size=BATCH, shuffle=train, num_workers=0,
    pin_memory=DEVICE.type == "cuda"
)
train_loader, val_loader, test_loader = (
    loader(train_p, train_y, True), loader(val_p, val_y), loader(test_p, test_y)
)

model = ResNet50BiLSTMAttention().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
best_f1, stale = -1.0, 0
best_path = OUT_DIR / "best.pt"

print("Device:", DEVICE, "videos:", len(paths))
for epoch in range(1, EPOCHS + 1):
    tr = run_epoch(model, train_loader, criterion, optimizer)
    va = run_epoch(model, val_loader, criterion)
    print(f"epoch {epoch:02d} train={tr} val={va}")
    if va["f1"] > best_f1:
        best_f1, stale = va["f1"], 0
        torch.save({
            "model": model.state_dict(),
            "cfg": {
                "data": {"classes": ["non_violence", "violence"], "num_frames": T, "image_size": SIZE},
                "model": {"backbone": "resnet50", "pretrained": False, "freeze_backbone": False,
                          "lstm_hidden": 512, "lstm_layers": 2, "bidirectional": True,
                          "dropout": 0.5, "use_attention": True, "num_classes": 2},
                "infer": {"window_frames": T, "stride": 8, "threshold": 0.6},
            },
            "val": va,
        }, best_path)
        print("saved", best_path)
    else:
        stale += 1
        if stale >= PATIENCE:
            print("early stopping")
            break

print("TEST:", run_epoch(model, test_loader, criterion))
print("BEST:", best_path)
