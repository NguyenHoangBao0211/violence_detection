"""Train one clean pipeline: ResNet50 + BiLSTM + temporal Attention.

Kaggle: change DATA_ROOT and optionally INIT_CHECKPOINT, then Run All.
"""
import copy, random
from pathlib import Path
import cv2, numpy as np
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

# ===== 1. CONFIG =====
DATA_ROOT = Path("/kaggle/input/datasets/daniazehra/rwf-2000/RWF-2000")
# Expected: train/Fight, train/NonFight, val/Fight, val/NonFight
OUT_DIR = Path("/kaggle/working/resnet50_bilstm_attention")
INIT_CHECKPOINT = None  # e.g. "/kaggle/input/.../best.pt"
SEED, T, SIZE = 42, 16, 224
BATCH, EPOCHS, PATIENCE = 2, 20, 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".webm"}
OUT_DIR.mkdir(parents=True, exist_ok=True)

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
seed_everything()

def videos(folder):
    return sorted(p for p in Path(folder).rglob("*")
                  if p.is_file() and p.suffix.lower() in EXTS)

def collect():
    # Keep the official RWF split; do not randomly mix train and val.
    train_f = videos(DATA_ROOT / "train" / "Fight")
    train_n = videos(DATA_ROOT / "train" / "NonFight")
    val_f = videos(DATA_ROOT / "val" / "Fight")
    val_n = videos(DATA_ROOT / "val" / "NonFight")
    if not all((train_f, train_n, val_f, val_n)):
        raise FileNotFoundError(
            f"Sai DATA_ROOT hoặc thiếu thư mục Fight/NonFight: {DATA_ROOT}")
    train_p, train_y = [str(p) for p in train_n + train_f], [0]*len(train_n)+[1]*len(train_f)
    val_p, val_y = [str(p) for p in val_n + val_f], [0]*len(val_n)+[1]*len(val_f)
    # Make test from validation only, preserving official train videos.
    val_p, test_p, val_y, test_y = train_test_split(
        val_p, val_y, test_size=.5, random_state=SEED, stratify=val_y)
    return train_p, train_y, val_p, val_y, test_p, test_y

MEAN, STD = (0.485, .456, .406), (.229, .224, .225)
train_tf = transforms.Compose([
    transforms.Resize((SIZE+32, SIZE+32)), transforms.RandomCrop(SIZE),
    transforms.RandomHorizontalFlip(), transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)])
eval_tf = transforms.Compose([
    transforms.Resize((SIZE, SIZE)), transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)])

def read_clip(path, train):
    cap = cv2.VideoCapture(str(path)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or n <= 0:
        cap.release(); raise RuntimeError(f"Không đọc được video: {path}")
    # Random contiguous clip only for train; fixed evenly-spaced clip for eval.
    start = random.randint(0, max(0, n-T)) if train else max(0, (n-T)//2)
    frames = []
    for idx in range(start, min(start+T, n)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, bgr = cap.read()
        if ok: frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames: raise RuntimeError(f"Không decode được video: {path}")
    while len(frames) < T: frames.append(frames[-1])
    tf = train_tf if train else eval_tf
    return torch.stack([tf(Image.fromarray(f)) for f in frames[:T]])

class VideoDataset(Dataset):
    def __init__(self, paths, labels, train=False):
        self.paths, self.labels, self.train = paths, labels, train
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return read_clip(self.paths[i], self.train), torch.tensor(self.labels[i], dtype=torch.long)

class TemporalAttention(nn.Module):
    def __init__(self, d):
        super().__init__(); self.score = nn.Sequential(nn.Linear(d,d//2), nn.Tanh(), nn.Linear(d//2,1))
    def forward(self, h):
        a = torch.softmax(self.score(h).squeeze(-1), dim=1)
        return torch.bmm(a.unsqueeze(1), h).squeeze(1), a

class ResNet50BiLSTMAttention(nn.Module):
    def __init__(self):
        super().__init__(); net = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(net.children())[:-1])
        self.lstm = nn.LSTM(2048, 512, 2, batch_first=True, bidirectional=True, dropout=.5)
        self.attention = TemporalAttention(1024)
        self.head = nn.Sequential(nn.Dropout(.5), nn.Linear(1024, 2))
    def forward(self, x):
        b,t,c,h,w = x.shape; z = self.cnn(x.reshape(b*t,c,h,w)).flatten(1)
        z,_ = self.lstm(z.view(b,t,2048)); context, alpha = self.attention(z)
        return self.head(context), alpha

def load_old(model, path):
    if not path: return model
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt); mapped = {}
    # Supports project checkpoints (spatial/temporal) and this notebook format.
    for k,v in state.items():
        k = k.replace("spatial.features.", "cnn.").replace("temporal.lstm.", "lstm.")
        k = k.replace("temporal.attn.", "attention.").replace("temporal.classifier.", "head.")
        mapped[k] = v
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    print("Loaded checkpoint:", path, "missing:", len(missing), "unexpected:", len(unexpected))
    return model

def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None; model.train(train); ys=[]; ps=[]; losses=[]
    with (torch.enable_grad() if train else torch.no_grad()):
        for x,y in tqdm(loader, leave=False):
            x,y = x.to(DEVICE), y.to(DEVICE)
            if train: optimizer.zero_grad(set_to_none=True)
            logits,_ = model(x); loss = criterion(logits,y)
            if train: loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            losses.append(loss.item()); ys += y.cpu().tolist(); ps += logits.argmax(1).cpu().tolist()
    return {"loss": float(np.mean(losses)), "accuracy": accuracy_score(ys,ps),
            "f1": f1_score(ys,ps,average="macro",zero_division=0)}

train_p, train_y, val_p, val_y, test_p, test_y = collect()
loader = lambda p,y,tr=False: DataLoader(VideoDataset(p,y,tr), batch_size=BATCH, shuffle=tr,
                                           num_workers=0, pin_memory=DEVICE.type=="cuda")
train_loader, val_loader, test_loader = loader(train_p,train_y,True), loader(val_p,val_y), loader(test_p,test_y)
model = load_old(ResNet50BiLSTMAttention().to(DEVICE), INIT_CHECKPOINT)
criterion = nn.CrossEntropyLoss(); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
best, stale, best_path = -1.0, 0, OUT_DIR / "best.pt"
print("Device:", DEVICE, "train/val/test:", len(train_p), len(val_p), len(test_p))
for epoch in range(1,EPOCHS+1):
    tr = run_epoch(model, train_loader, criterion, optimizer); va = run_epoch(model,val_loader,criterion)
    print(f"epoch {epoch:02d} train={tr} val={va}")
    if va["f1"] > best:
        best, stale = va["f1"], 0
        torch.save({"model":model.state_dict(), "cfg":{"data":{"classes":["non_violence","violence"],"num_frames":T,"image_size":SIZE},
            "model":{"backbone":"resnet50","pretrained":False,"freeze_backbone":False,"lstm_hidden":512,"lstm_layers":2,"bidirectional":True,"dropout":.5,"use_attention":True,"num_classes":2},
            "infer":{"window_frames":T,"stride":8,"threshold":.6}}, "val":va}, best_path)
        print("saved", best_path)
    else:
        stale += 1
        if stale >= PATIENCE: break
print("TEST:", run_epoch(model,test_loader,criterion)); print("BEST:", best_path)
