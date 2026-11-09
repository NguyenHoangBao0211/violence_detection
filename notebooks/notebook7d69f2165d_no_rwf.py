# %% Cell 0
"""Train one clean pipeline: ResNet50 + BiLSTM + temporal Attention.

Kaggle: change the dataset roots below and optionally INIT_CHECKPOINT, then Run All.
"""
import copy, random, re, warnings, hashlib
from pathlib import Path
import cv2, numpy as np, pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights


# %% Cell 1
from pathlib import Path
import random
import numpy as np
import torch

# =========================
# PATH
# =========================

AIRTLAB_ROOT = Path(
    "/kaggle/input/datasets/hoangbao2004/datawebcam/A-Dataset-for-Automatic-Violence-Detection-in-Videos-master/violence-detection-dataset"
)

def find_rlvs_root():
    preferred = Path("/kaggle/input/datasets/mohamedmustafa/real-life-violence-situations-dataset/Real Life Violence Dataset")
    if (preferred / "NonViolence").is_dir() and (preferred / "Violence").is_dir():
        return preferred
    for folder in sorted(Path("/kaggle/input").rglob("NonViolence")):
        if folder.is_dir() and (folder.parent / "Violence").is_dir():
            return folder.parent
    raise FileNotFoundError("Add Real Life Violence Situations Dataset: need NonViolence/ and Violence/.")

RLVS_ROOT = find_rlvs_root()

# Merged (messaoudiriham/merged-violence-dataset, 7000 video) KHONG dung nua:
# lon nhat, trung nhieu voi RLVS, va lan at clip webcam tu quay (MyWebcam).

def find_datasetwebcam_root():
    # Dataset Kaggle ten DatasetWebCam. Ben trong la AIRTLab:
    # violence-detection-dataset/non-violent|violent/cam1|cam2
    named = []
    for path in Path("/kaggle/input").rglob("*"):
        if path.is_dir() and path.name.lower().replace("_", "").replace("-", "") in {"datasetwebcam", "datawebcam"}:
            named.append(path)
    for root in named:
        for sub in root.rglob("violence-detection-dataset"):
            if (sub / "non-violent").is_dir() and (sub / "violent").is_dir():
                return sub
        if (root / "non-violent").is_dir() and (root / "violent").is_dir():
            return root
    for sub in Path("/kaggle/input").rglob("violence-detection-dataset"):
        if (sub / "non-violent").is_dir() and (sub / "violent").is_dir():
            return sub
    fallback = Path(
        "/kaggle/input/datasets/bohong111/datawebcam/A-Dataset-for-Automatic-Violence-Detection-in-Videos-master/violence-detection-dataset"
    )
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Khong tim thay DatasetWebCam. Hay Add dataset do vao notebook.")

DATASET_WEBCAM_ROOT = find_datasetwebcam_root()

def find_mywebcam_root():
    # Clip webcam tu quay, cat 1 s/clip: <root>/non_violence|violence/<video>_<NNN>.mp4
    # (khac AIRTLab: AIRTLab dung non-violent|violent co gach ngang).
    preferred = Path("/kaggle/input/datasets/hoangbao2004/datacamera1/DataWebCam")
    if (preferred / "non_violence").is_dir() and (preferred / "violence").is_dir():
        return preferred
    for path in Path("/kaggle/input").rglob("non_violence"):
        if path.is_dir() and (path.parent / "violence").is_dir():
            return path.parent
    print("CANH BAO: khong tim thay clip webcam tu quay (non_violence/ + violence/). Hay Add dataset do vao Input.")
    return Path("/kaggle/input/mywebcam-khong-co")

MYWEBCAM_ROOT = find_mywebcam_root()


if not AIRTLAB_ROOT.exists():
    print("AIRTLab cu khong co trong Input. Dung DatasetWebCam:", DATASET_WEBCAM_ROOT)
    AIRTLAB_ROOT = DATASET_WEBCAM_ROOT

# Not used: the model needs 16 consecutive frames of one video.
#   karandeep98/real-life-violence-and-nonviolence-data : loose .jpg frames, not grouped by video
#   quangkhanh205/airtlab-datasets                      : AIRTLab again as extracted frames (duplicate)

def find_init_checkpoint():
    # Model tren Kaggle: Input -> Models -> hoangbao2004/modelv1 (hoac modelv3 cu).
    preferred = Path("/kaggle/input/models/hoangbao2004/modelv1/pytorch/default/1/best.pt")
    if preferred.is_file():
        return preferred
    candidates = []
    for base in (Path("/kaggle/input/models"), Path("/kaggle/input")):
        if not base.is_dir():
            continue
        for name in ("modelv1", "modelv3"):
            for folder in base.rglob(name):
                if folder.is_dir():
                    candidates.extend(p for p in folder.rglob("*.pt") if p.is_file())
        if candidates:
            break
    if not candidates:
        # Chi can khi INIT_FROM = "checkpoint" (kiem tra o cell EXPERIMENT).
        print("Khong tim thay checkpoint (.pt): chi train duoc voi INIT_FROM = 'imagenet'.")
        return None
    preferred_names = {"best.pt", "bestv1.pt", "bestv3.pt"}
    return sorted(candidates, key=lambda p: (p.name.lower() not in preferred_names, len(p.name), str(p)))[0]


# Input modelYolo11: find weights without assuming owner/framework/version.
# Set an exact path here if the Input contains several different checkpoints.
YOLO_CHECKPOINT_OVERRIDE = None

def find_yolo_checkpoint():
    if YOLO_CHECKPOINT_OVERRIDE:
        path = Path(YOLO_CHECKPOINT_OVERRIDE)
        if not path.is_file(): raise FileNotFoundError(path)
        return path
    candidates = sorted(p for p in Path("/kaggle/input").rglob("*.pt")
                        if any("yolo11" in part.lower() for part in p.parts))
    if len(candidates) > 1:
        print("Multiple YOLO11 weights; set YOLO_CHECKPOINT_OVERRIDE:")
        for path in candidates: print("  ", path)
        return None
    return candidates[0] if candidates else None

YOLO_CHECKPOINT = find_yolo_checkpoint()
# YOLO weights are separate from CNN-BiLSTM weights; do not pass them to load_old.
INIT_CHECKPOINT = find_init_checkpoint()

OUT_DIR = Path(
    "/kaggle/working/resnet50_bilstm_attention"
)

MANIFEST_PATH = OUT_DIR / "manifest.csv"

# =========================
# CONFIG
# =========================

# 'combined' gop AIRTLab (DataWebCam) + RLVS + MyWebcam (DataCamera1).
DATA_MODE = "combined"
EVAL_DOMAIN = "external"  # public manifest checks; fine-tuning evaluates MyWebcam separately
# read_clip lay 16 frame LIEN TIEP (~0.5 s o 30 fps), giong webcam va best2.pt.
# Video ngan hon khoang nay bi loai; 2.25 s cu loai oan cac clip AIRTLab ~2 s.
CLIP_SPAN_SECONDS = (16 - 1) / 30

# SEED chi dung de CHIA train/val/test (manifest). Giu 42 de moi lan chay cung tap test;
# seed huan luyen la TRAIN_SEED o cell EXPERIMENT.
SEED = 42
T = 16
SIZE = 224
BATCH = 2
EPOCHS = 20
PATIENCE = 5

CFG = {"seed": SEED}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


# =========================
# CHECK
# =========================

print("===== KIEM TRA CAU HINH =====")

print("Device:", DEVICE)
print("DATA_MODE:", DATA_MODE)

REQUIRED_ROOTS = {
    "combined": [("AIRTLab", AIRTLAB_ROOT), ("RLVS", RLVS_ROOT), ("DatasetWebCam", DATASET_WEBCAM_ROOT), ("MyWebcam", MYWEBCAM_ROOT)],
    "airtlab": [("AIRTLab", AIRTLAB_ROOT)],
    "manifest": [],
}[DATA_MODE]

print("\nDataset:")
present = []
seen = set()
for name, root in REQUIRED_ROOTS:
    print(f"  {name}: {root} | ton tai: {root.exists()}")
    if not root.exists():
        print(f"  Bo qua {name}: khong co trong Input.")
        continue
    key = str(root.resolve())
    if key in seen:
        print(f"  Bo qua {name}: trung thu muc da dung.")
        continue
    seen.add(key)
    present.append((name, root))
if not present:
    raise FileNotFoundError("Khong co dataset nao trong Input.")
REQUIRED_ROOTS = present

print("\nCheckpoint:")
print(INIT_CHECKPOINT)
print("Checkpoint ton tai:", bool(INIT_CHECKPOINT and INIT_CHECKPOINT.exists()))

print("\nYOLO11 Input weights:", YOLO_CHECKPOINT)
if YOLO_CHECKPOINT is None:
    print("YOLO11 weights missing or ambiguous; base/fine-tuning does not require YOLO.")
print("\nOK - Dataset da tim thay.")
print("Co the chay cell tiep theo.")


# %% Cell 2
# =========================
# EXPERIMENT (ablation)
# =========================
# Chay notebook mot lan cho moi thi nghiem; ket qua o OUT_DIR/<EXPERIMENT>/.
# Manifest (split train/val/test) dung chung o OUT_DIR nen moi thi nghiem
# duoc danh gia tren cung mot tap test.
#   input: "diff" = hieu frame (Sudhakaran & Lanz 2017), "rgb" = frame goc
#   arch : "single" = ResNet50-(Bi)LSTM, "two_stream" = RGB ResNet50 + motion ResNet18
EXPERIMENTS = {
    "diff":         dict(input="diff", arch="single",     bidirectional=True,  attention=True,  frames=16),  # mo hinh hien tai
    "rgb":          dict(input="rgb",  arch="single",     bidirectional=True,  attention=True,  frames=16),
    "two_stream":   dict(input="rgb",  arch="two_stream", bidirectional=True,  attention=True,  frames=16),
    "no_attention": dict(input="diff", arch="single",     bidirectional=True,  attention=False, frames=16),
    "lstm":         dict(input="diff", arch="single",     bidirectional=False, attention=True,  frames=16),
    "diff_t32":     dict(input="diff", arch="single",     bidirectional=True,  attention=True,  frames=32),
    # RQ1/RQ2 tren RGB: chi khong gian (khong LSTM), LSTM 1 chieu, BiLSTM khong attention.
    "spatial_only":     dict(input="rgb", arch="single", temporal="none", bidirectional=True,  attention=False, frames=16),
    "rgb_lstm":         dict(input="rgb", arch="single",                  bidirectional=False, attention=True,  frames=16),
    "rgb_no_attention": dict(input="rgb", arch="single",                  bidirectional=True,  attention=False, frames=16),
    # Model cho webcam (CPU, thoi gian thuc): MobileNetV2 thay ResNet50, it hon ~13 lan FLOPs moi frame.
    "mobilenet":        dict(input="rgb", arch="single", backbone="mobilenet_v2", bidirectional=True, attention=True, frames=16),
}
EXPERIMENT = "mobilenet"  # model webcam; doi sang "diff", "rgb", ... cho cac thi nghiem khac
EXP = EXPERIMENTS[EXPERIMENT]
T = EXP["frames"]
if T > 16:
    BATCH = 1  # 32 frame x ResNet50 khong vua batch 2 tren GPU Kaggle
elif EXP.get("backbone") == "mobilenet_v2":
    BATCH = 8  # MobileNetV2 nhe hon nhieu: batch lon hon, train nhanh hon

# "checkpoint": khoi tao tu INIT_CHECKPOINT (nhu truoc).
# "imagenet"  : chi dung trong so ImageNet. Dung cho BANG ABLATION de moi bien the
#               xuat phat nhu nhau; checkpoint cu da duoc train dung kien truc "diff"
#               nen se thien vi bien the do.
INIT_FROM = "imagenet"
if INIT_FROM == "checkpoint" and not (INIT_CHECKPOINT and Path(INIT_CHECKPOINT).exists()):
    raise FileNotFoundError("INIT_FROM = 'checkpoint' can Add ModelIV3 (modelv3/*.pt) vao Input.")

# Seed huan luyen. Bang nghien cuu: chay moi thi nghiem voi 42, 43, 44 roi bao cao mean +- std.
TRAIN_SEED = 42
seed_everything(TRAIN_SEED)

# Cross-dataset: vd {"rlvs", "merged"} -> train/val chi tren cac nguon nay,
# test van tren tat ca nguon; bang theo nguon danh dau in-domain / cross.
TRAIN_SOURCES = None

RUN_DIR = OUT_DIR / f"{EXPERIMENT}_s{TRAIN_SEED}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
print("EXPERIMENT:", EXPERIMENT, EXP, "| TRAIN_SEED:", TRAIN_SEED, "| INIT_FROM:", INIT_FROM, "| TRAIN_SOURCES:", TRAIN_SOURCES)
print("RUN_DIR:", RUN_DIR)

# Two-stage training: public videos first, webcam adaptation second.
BASE_EPOCHS = EPOCHS
FINETUNE_EPOCHS = 10
FREEZE_EPOCHS = 2
FINETUNE_PATIENCE = 4
BASE_SOURCES = {"airtlab", "datasetwebcam", "rlvs"}
WEBCAM_SOURCES = {"mywebcam"}
# Optional compatible base checkpoint: skip base training when supplied.
BASE_CHECKPOINT = None
# Public replay count = this fraction of webcam training count.
REPLAY_RATIO = 0.25
BASE_LR = 1e-4
FINETUNE_HEAD_LR = 1e-5
FINETUNE_CNN_LR = 1e-6
RUN_DIR = OUT_DIR / f"{EXPERIMENT}_s{TRAIN_SEED}_webcam_finetune"
RUN_DIR.mkdir(parents=True, exist_ok=True)
print("Two-stage outputs:", RUN_DIR)


# %% Cell 3
def build_airtlab_manifest(data_root, output_csv, seed):
    """Group both cameras of each class/clip ID before stratified splitting."""
    data_root, output_csv = Path(data_root).resolve(), Path(output_csv)
    rows = []
    for class_name, label in [('non-violent', 0), ('violent', 1)]:
        for camera in ('cam1', 'cam2'):
            folder = data_root / class_name / camera
            if not folder.is_dir():
                raise FileNotFoundError(f'Thi\u1ebfu th\u01b0 m\u1ee5c AIRTLab: {folder}')
            videos = sorted(p for p in folder.iterdir()
                            if p.is_file() and p.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv'})
            if not videos:
                raise ValueError(f'Kh\u00f4ng c\u00f3 video trong {folder}')
            for video in videos:
                if not video.stem.isdecimal():
                    raise ValueError(f'AIRTLab g\u1ed1c d\u00f9ng t\u00ean s\u1ed1, kh\u00f4ng t\u1ef1 \u0111o\u00e1n c\u1eb7p camera: {video.name}')
                cap = cv2.VideoCapture(str(video))
                try:
                    fps = float(cap.get(cv2.CAP_PROP_FPS))
                    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    ok, _ = cap.read()
                    if not cap.isOpened() or not ok or not np.isfinite(fps) or fps <= 0 or count <= 0:
                        raise ValueError(f'Kh\u00f4ng \u0111\u1ecdc \u0111\u01b0\u1ee3c video/FPS: {video}. Ki\u1ec3m tra Input c\u00f3 video th\u1eadt.')
                finally:
                    cap.release()
                rows.append({'path': video.relative_to(data_root).as_posix(), 'label': label,
                             'group_id': f'{class_name}/{int(video.stem)}',
                             'domain': 'external', 'camera': camera, 'last_time': (count - 1) / fps})
    df = pd.DataFrame(rows)
    for group, subset in df.groupby('group_id'):
        if len(subset) != 2 or set(subset.camera) != {'cam1', 'cam2'}:
            raise ValueError(f'C\u1eb7p camera thi\u1ebfu/tr\u00f9ng: {group}')

    # Co clip AIRTLab chi dai 2s: loai CA NHOM neu mot goc khong du 2.25s.
    # Khong keo gian thoi gian hoac lap frame de che viec thieu chuyen dong.
    short_groups = set(df.loc[df.last_time + 1e-6 < CLIP_SPAN_SECONDS, 'group_id'])
    excluded = df[df.group_id.isin(short_groups)].copy()
    excluded['reason'] = 'camera_pair_shorter_than_required_clip_span'
    df = df[~df.group_id.isin(short_groups)].copy()
    rng = np.random.default_rng(seed)
    assignments = {}
    for label in (0, 1):
        groups = sorted(df.loc[df.label == label, 'group_id'].unique())
        if len(groups) < 3:
            raise ValueError(f'L\u1edbp {label} c\u1ea7n \u00edt nh\u1ea5t 3 c\u1eb7p camera h\u1ee3p l\u1ec7 \u0111\u1ec3 chia train/val/test.')
        groups = list(rng.permutation(groups))
        n_val = max(1, round(len(groups) * 0.15))
        n_test = max(1, round(len(groups) * 0.15))
        n_train = len(groups) - n_val - n_test
        for split, selected in [('train', groups[:n_train]),
                                ('val', groups[n_train:n_train + n_val]),
                                ('test', groups[n_train + n_val:])]:
            assignments.update({group: split for group in selected})
    df['split'] = df.group_id.map(assignments)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    excluded.to_csv(output_csv.parent / 'excluded_airtlab.csv', index=False)
    df[['path', 'label', 'group_id', 'split', 'domain']].to_csv(output_csv, index=False)
    print(f'AIRTLab: {len(df)} video, {df.group_id.nunique()} c\u1eb7p camera; lo\u1ea1i {len(excluded)} video ng\u1eafn/c\u00f9ng c\u1eb7p.')
    print('Manifest:', output_csv)
    return df


def build_rlvs_manifest(data_root, output_csv, seed):
    """RLVS khong co cap camera/split goc: moi video la mot nhom, chia nhu AIRTLab."""
    data_root, output_csv = Path(data_root).resolve(), Path(output_csv)
    rows = []
    for folder_name, label in (('NonViolence', 0), ('Violence', 1)):
        folder = data_root / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f'Thi\u1ebfu th\u01b0 m\u1ee5c RLVS: {folder}')
        videos = sorted(p for p in folder.iterdir()
                        if p.is_file() and p.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv'})
        if not videos:
            raise ValueError(f'Kh\u00f4ng c\u00f3 video trong {folder}')
        for video in videos:
            rows.append({'path': video.relative_to(data_root).as_posix(), 'label': label,
                         'group_id': f'{folder_name}/{video.stem}', 'domain': 'external'})
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    assignments = {}
    for label in (0, 1):
        groups = sorted(df.loc[df.label == label, 'group_id'].unique())
        if len(groups) < 3:
            raise ValueError(f'L\u1edbp {label} c\u1ea7n \u00edt nh\u1ea5t 3 video RLVS \u0111\u1ec3 chia train/val/test.')
        groups = list(rng.permutation(groups))
        n_val = max(1, round(len(groups) * 0.15))
        n_test = max(1, round(len(groups) * 0.15))
        n_train = len(groups) - n_val - n_test
        for split, selected in [('train', groups[:n_train]),
                                ('val', groups[n_train:n_train + n_val]),
                                ('test', groups[n_train + n_val:])]:
            assignments.update({group: split for group in selected})
    df['split'] = df.group_id.map(assignments)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df[['path', 'label', 'group_id', 'split', 'domain']].to_csv(output_csv, index=False)
    print(f'{data_root.name}: {len(df)} video, {df.group_id.nunique()} nh\u00f3m.')
    return df


def build_mywebcam_manifest(data_root, output_csv, seed):
    """Clip 1 s cat tu video webcam dai: <root>/non_violence|violence/<video>_<NNN>.mp4.

    Cac giay lien nhau gan nhu giong het, chia ngau nhien tung clip se ro ri test.
    Chia theo doan thoi gian trong MOI video: 70% giay dau -> train, 15% -> val,
    15% cuoi -> test, bo 2 s o moi ranh gioi. Giong split_of() trong
    src/data/webcam_dataset.py va notebooks/finetune_webcam.py.
    """
    data_root, output_csv = Path(data_root).resolve(), Path(output_csv)
    rows = []
    for folder_name, label in (('non_violence', 0), ('violence', 1)):
        folder = data_root / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f'Thieu thu muc webcam: {folder}')
        by_video = {}
        for clip in sorted(folder.glob('*.mp4')):
            m = re.match(r'^(.*)_(\d{3})$', clip.stem)
            if m is None:
                raise ValueError(f'Ten clip webcam phai la <video>_<NNN>.mp4: {clip.name}')
            by_video.setdefault(m.group(1), []).append((int(m.group(2)), clip))
        for video, clips in by_video.items():
            n = len(clips)
            val_start, test_start = round(n * 0.70), round(n * 0.85)
            for idx, clip in clips:
                if idx < val_start:
                    split = 'train'
                elif val_start + 2 <= idx < test_start:
                    split = 'val'
                elif idx >= test_start + 2:
                    split = 'test'
                else:
                    continue  # giay ranh gioi: khong dung
                rows.append({'path': clip.relative_to(data_root).as_posix(), 'label': label,
                             'group_id': f'{folder_name}/{video}/{split}', 'split': split, 'domain': 'webcam'})
    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df[['path', 'label', 'group_id', 'split', 'domain']].to_csv(output_csv, index=False)
    print(f'MyWebcam: {len(df)} clip 1 s')
    print(df.groupby(['split', 'label']).size().to_string())
    return df


def validate_manifest(csv_path, data_root=None, roots=None):
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f'Kh\u00f4ng t\u00ecm th\u1ea5y manifest: {csv_path}. '
                           'V\u1edbi AIRTLab/RLVS, ch\u1ea1y tr\u1ecdn cell t\u1ea1o v\u00e0 ki\u1ec3m tra manifest; '
                           'v\u1edbi ch\u1ebf \u0111\u1ed9 manifest, ki\u1ec3m tra \u0111\u01b0\u1eddng d\u1eabn CSV \u0111\u00e3 chu\u1ea9n b\u1ecb.')
    df = pd.read_csv(csv_path, dtype={'path': str, 'group_id': str, 'split': str, 'domain': str, 'source': str})
    required = {'path', 'label', 'group_id', 'split', 'domain'}
    if not required.issubset(df.columns) or df.empty:
        raise ValueError(f'Manifest ph\u1ea3i c\u00f3 d\u1eef li\u1ec7u v\u00e0 c\u00e1c c\u1ed9t {sorted(required)}')
    if roots is not None and 'source' not in df.columns:
        raise ValueError('Manifest g\u1ed9p nhi\u1ec1u dataset c\u1ea7n c\u1ed9t "source" \u0111\u1ec3 bi\u1ebft root g\u1ed1c c\u1ee7a m\u1ed7i video.')
    check_cols = list(required) + (['source'] if roots is not None else [])
    if df[check_cols].isna().any().any():
        raise ValueError('Manifest c\u00f3 \u00f4 tr\u1ed1ng.')
    for col in check_cols:
        if col == 'label':
            continue
        df[col] = df[col].str.strip()
        if df[col].eq('').any():
            raise ValueError(f'C\u1ed9t {col} c\u00f3 gi\u00e1 tr\u1ecb tr\u1ed1ng.')
    if not df.label.isin([0, 1]).all():
        raise ValueError('label ch\u1ec9 nh\u1eadn 0 ho\u1eb7c 1.')
    df['label'] = df.label.astype(int)
    if not df.split.isin(['train', 'val', 'test']).all():
        raise ValueError('split ch\u1ec9 nh\u1eadn train/val/test.')
    if not df.domain.isin(['webcam', 'external']).all():
        raise ValueError('domain ch\u1ec9 nh\u1eadn webcam/external.')
    if roots is not None:
        missing_sources = set(df.source) - set(roots)
        if missing_sources:
            raise ValueError(f'Kh\u00f4ng c\u00f3 root cho source: {sorted(missing_sources)}')
    resolved = []
    for row in df.itertuples():
        if roots is not None:
            base = Path(roots[row.source]).resolve()
        elif data_root is not None:
            base = Path(data_root).resolve()
        else:
            base = csv_path.parent.resolve()
        rel = Path(row.path)
        path = (base / rel).resolve()
        if rel.is_absolute() or not path.is_relative_to(base):
            raise ValueError(f'path ph\u1ea3i t\u01b0\u01a1ng \u0111\u1ed1i v\u00e0 n\u1eb1m trong dataset: {row.path}')
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved.append(str(path))
    df['resolved_path'] = resolved
    if df.resolved_path.duplicated().any():
        raise ValueError('M\u1ed9t file xu\u1ea5t hi\u1ec7n nhi\u1ec1u l\u1ea7n trong manifest.')
    if (df.groupby('group_id').split.nunique() > 1).any():
        raise ValueError('R\u00f2 r\u1ec9 d\u1eef li\u1ec7u: c\u00f9ng group_id xu\u1ea5t hi\u1ec7n \u1edf nhi\u1ec1u split.')
    hashes = []
    for path in tqdm(df.resolved_path, desc='Ki\u1ec3m tra file tr\u00f9ng'):
        digest = hashlib.sha256()
        with open(path, 'rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        hashes.append(digest.hexdigest())
    df['sha256'] = hashes
    dup_cols = [c for c in ('source', 'split', 'domain', 'path', 'label', 'sha256') if c in df.columns]
    if df.sha256.duplicated(keep=False).any():
        # Video trung noi dung (thuong gap trong RLVS goc): xu ly tu dong thay vi dung han.
        all_dups = df.loc[df.sha256.duplicated(keep=False)].sort_values('sha256')

        # 1) Trung noi dung nhung nhan mau thuan (vd 1 ban ghi Violence, ban kia NonViolence):
        # khong the tu quyet nhan nao dung -> loai ca nhom, ghi ra file de kiem tra thu cong.
        conflict_mask = all_dups.groupby('sha256').label.transform('nunique') > 1
        conflict_hashes = set(all_dups.loc[conflict_mask, 'sha256'])
        if conflict_hashes:
            conflict_rows = df.loc[df.sha256.isin(conflict_hashes), dup_cols].sort_values('sha256')
            print(f'CANH BAO: {len(conflict_rows)} video trung noi dung nhung nhan mau thuan; loai ca nhom:')
            print(conflict_rows.to_string(index=False))
            conflict_rows.to_csv(csv_path.parent / 'label_conflict_duplicates.csv', index=False)
            df = df[~df.sha256.isin(conflict_hashes)].copy()

        # 2) Trung noi dung, cung nhan, khac split (vd 1 ban o train, 1 ban o val/test):
        # giu dung 1 ban/nhom, uu tien giu ban o test > val > train de bao ve tap danh gia
        # (tranh video train roi vo tinh xuat hien lai o val/test lam sai lech metric).
        split_priority = {'test': 0, 'val': 1, 'train': 2}
        dup_mask = df.sha256.duplicated(keep=False)
        if dup_mask.any():
            dup_group = df.loc[dup_mask].copy()
            dup_group['_prio'] = dup_group.split.map(split_priority)
            keep_idx = dup_group.sort_values(['sha256', '_prio']).groupby('sha256').head(1).index
            drop_idx = dup_group.index.difference(keep_idx)
            dropped = df.loc[drop_idx, dup_cols].sort_values('sha256')
            print(f'Loai {len(drop_idx)} video trung noi dung (giu 1 ban/nhom, uu tien test > val > train):')
            print(dropped.to_string(index=False))
            dropped.to_csv(csv_path.parent / 'duplicate_videos_removed.csv', index=False)
            df = df.drop(index=drop_idx).copy()

        df = df.reset_index(drop=True)
    for split in ('train', 'val', 'test'):
        target = df[(df.split == split) & (df.domain == EVAL_DOMAIN)]
        if set(target.label) != {0, 1}:
            raise ValueError(f'{split} {EVAL_DOMAIN} c\u1ea7n \u0111\u1ee7 2 l\u1edbp \u0111\u1ec3 train/\u0111\u00e1nh gi\u00e1 \u0111\u00fang m\u1ee5c ti\u00eau.')
        if target.group_id.nunique() < 3:
            warnings.warn(f'{split} {EVAL_DOMAIN} c\u00f3 d\u01b0\u1edbi 3 nh\u00f3m; s\u1ed1 li\u1ec7u c\u00f3 th\u1ec3 dao \u0111\u1ed9ng m\u1ea1nh.')
    return df



def prepare_manifest():
    # Cung cell dinh nghia builder va validator: khong phu thuoc cell tao CSV rieng.
    # Giu manifest da ton tai de khong tu thay doi split khi chay lai cell.
    if DATA_MODE == 'combined' and not Path(MANIFEST_PATH).is_file():
        print('Ch\u01b0a c\u00f3 manifest; \u0111ang g\u1ed9p AIRTLab + RLVS + MyWebcam...')
        parts = []
        for source, builder, root in (
            ('airtlab', build_airtlab_manifest, AIRTLAB_ROOT),
            ('rlvs', build_rlvs_manifest, RLVS_ROOT),
            # DatasetWebCam chua AIRTLab: non-violent/violent + cam1/cam2
            ('datasetwebcam', build_airtlab_manifest, DATASET_WEBCAM_ROOT),
            # Clip webcam tu quay (1 s/clip), chia theo doan thoi gian trong moi video
            ('mywebcam', build_mywebcam_manifest, MYWEBCAM_ROOT),
        ):
            if not Path(root).exists():
                print('Bo qua', source, '- khong co trong Input')
                continue
            if source == 'datasetwebcam' and Path(root).resolve() == Path(AIRTLAB_ROOT).resolve():
                print('Bo qua datasetwebcam - trung AIRTLab')
                continue
            part = builder(root, OUT_DIR / f'manifest_{source}.csv', CFG['seed']).copy()
            part['source'] = source
            part['group_id'] = source + '/' + part['group_id'].astype(str)
            parts.append(part)
        combined = pd.concat(parts, ignore_index=True)
        Path(MANIFEST_PATH).parent.mkdir(parents=True, exist_ok=True)
        combined[['path', 'label', 'group_id', 'split', 'domain', 'source']].to_csv(MANIFEST_PATH, index=False)
    if DATA_MODE == 'airtlab' and not Path(MANIFEST_PATH).is_file():
        print('Ch\u01b0a c\u00f3 manifest; \u0111ang t\u1ea1o t\u1eeb AIRTLab:', AIRTLAB_ROOT)
        build_airtlab_manifest(AIRTLAB_ROOT, MANIFEST_PATH, CFG['seed'])
    if DATA_MODE == 'combined':
        print('\u0110\u00e1nh gi\u00e1 tr\u00ean AIRTLab + RLVS + MyWebcam g\u1ed9p (external), ch\u01b0a ph\u1ea3i \u0111\u1ed9 ch\u00ednh x\u00e1c webcam laptop.')
    elif DATA_MODE == 'airtlab':
        print('\u0110\u00e1nh gi\u00e1 tr\u00ean AIRTLab (external), ch\u01b0a ph\u1ea3i \u0111\u1ed9 ch\u00ednh x\u00e1c webcam th\u1ef1c t\u1ebf.')
    roots = {name: path for name, path in (
        ('airtlab', AIRTLAB_ROOT), ('rlvs', RLVS_ROOT),
        ('datasetwebcam', DATASET_WEBCAM_ROOT),
        ('mywebcam', MYWEBCAM_ROOT),
    ) if Path(path).exists()} if DATA_MODE == 'combined' else None
    data_root = AIRTLAB_ROOT if DATA_MODE == 'airtlab' else None
    return validate_manifest(MANIFEST_PATH, data_root=data_root, roots=roots)


if Path(MANIFEST_PATH).is_file() and DATA_MODE == 'combined':
    old = pd.read_csv(MANIFEST_PATH)
    needed = {'rlvs'} | ({'mywebcam'} if Path(MYWEBCAM_ROOT).exists() else set())
    if 'source' not in old.columns or not needed <= set(old.source) or bool({'merged', 'rwf2000'} & set(old.source)):
        print('Manifest cu thieu RLVS/MyWebcam hoac con nguon da bo -> tao lai.')
        Path(MANIFEST_PATH).unlink()
manifest = prepare_manifest()
print(manifest.groupby(['source', 'split', 'label']).size().rename('videos').to_string())
print(manifest.groupby(['split', 'domain', 'label']).size().rename('videos').to_string())
manifest.to_csv(OUT_DIR / 'manifest_resolved.csv', index=False)


# %% Cell 5
def collect():
    # Nguon du lieu: manifest da gop AIRTLab + RLVS + MyWebcam (hoac 1 dataset tuy DATA_MODE),
    # duoc tao/kiem tra o cell truoc (bien `manifest`, cot `resolved_path`).
    print("===== DATASET (manifest) =====")
    print(manifest.groupby(["split", "label"]).size().to_string())

    def split_xy(name):
        subset = manifest[manifest.split == name]
        # Cross-dataset: train/val chi lay nguon trong TRAIN_SOURCES, test giu tat ca.
        if TRAIN_SOURCES and name != "test":
            if "source" not in subset.columns:
                raise ValueError("TRAIN_SOURCES can DATA_MODE='combined' (manifest co cot source).")
            subset = subset[subset.source.isin(TRAIN_SOURCES)]
        return subset.resolved_path.tolist(), subset.label.astype(int).tolist()

    train_p, train_y = split_xy("train")
    val_p, val_y = split_xy("val")
    test_p, test_y = split_xy("test")

    print("\n===== SPLIT =====")
    print("Train:", len(train_p))
    print("Val  :", len(val_p))
    print("Test :", len(test_p))

    print("\nTrain:")
    print("  Non-violent:", train_y.count(0))
    print("  Violent    :", train_y.count(1))

    print("\nVal:")
    print("  Non-violent:", val_y.count(0))
    print("  Violent    :", val_y.count(1))

    print("\nTest:")
    print("  Non-violent:", test_y.count(0))
    print("  Violent    :", test_y.count(1))

    return (
        train_p,
        train_y,
        val_p,
        val_y,
        test_p,
        test_y
    )


# %% Cell 7
import torchvision.transforms.functional as TF

MEAN, STD = (0.485, .456, .406), (.229, .224, .225)

def clip_transform(frames, train):
    """Resize/crop/flip with ONE set of random parameters for the whole clip.

    Per-frame RandomCrop/RandomHorizontalFlip would shift and mirror each frame
    differently, which is fake motion between frames: it corrupts the LSTM
    input and the two-stream model's frame differences.
    """
    imgs = [Image.fromarray(f) for f in frames]
    if train:
        imgs = [TF.resize(im, [SIZE + 32, SIZE + 32]) for im in imgs]
        top, left = random.randint(0, 32), random.randint(0, 32)
        imgs = [TF.crop(im, top, left, SIZE, SIZE) for im in imgs]
        if random.random() < 0.5:
            imgs = [TF.hflip(im) for im in imgs]
    else:
        imgs = [TF.resize(im, [SIZE, SIZE]) for im in imgs]
    return torch.stack([TF.normalize(TF.to_tensor(im), MEAN, STD) for im in imgs])


# %% Cell 8
def frame_diffs(frames):
    """Frame 0 is black. Each later frame is the absolute difference from the previous one."""
    diffs = [np.zeros_like(frames[0])]
    for prev, cur in zip(frames, frames[1:]):
        diffs.append(cv2.absdiff(cur, prev))
    return diffs

# E5 do ben: cell DO BEN dat PERTURB truoc khi danh gia test, roi tra ve {}.
#   fps_step  : lay 1 frame moi k frame (30 fps -> 30/k fps)
#   jpeg      : nen JPEG chat luong q (nhu trang Camera gui frame)
#   brightness: nhan do sang voi he so b
PERTURB = {}

def perturb(frames):
    if PERTURB.get("jpeg"):
        q = [cv2.IMWRITE_JPEG_QUALITY, int(PERTURB["jpeg"])]
        frames = [cv2.imdecode(cv2.imencode(".jpg", f, q)[1], cv2.IMREAD_UNCHANGED) for f in frames]
    if PERTURB.get("brightness"):
        frames = [np.clip(f.astype(np.float32) * PERTURB["brightness"], 0, 255).astype(np.uint8) for f in frames]
    return frames

def read_clip(path, train):
    cap = cv2.VideoCapture(str(path)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or n <= 0:
        cap.release(); raise RuntimeError(f"Không đọc được video: {path}")
    # Random contiguous clip only for train; fixed evenly-spaced clip for eval.
    step = 1 if train else int(PERTURB.get("fps_step", 1))
    span = T * step
    start = random.randint(0, max(0, n-span)) if train else max(0, (n-span)//2)
    frames = []
    for idx in range(start, min(start+span, n), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, bgr = cap.read()
        if ok: frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames: raise RuntimeError(f"Không decode được video: {path}")
    while len(frames) < T: frames.append(frames[-1])
    frames = frames[:T]
    # Nhieu dat TRUOC frame_diffs: webcam nen JPEG anh goc, roi model moi tinh hieu frame.
    if not train and PERTURB:
        frames = perturb(frames)
    # two_stream nhan RGB va tu tinh hieu frame ben trong model.
    if EXP["input"] == "diff":
        frames = frame_diffs(frames)
    return clip_transform(frames, train)

class VideoDataset(Dataset):
    def __init__(self, paths, labels, train=False):
        self.paths, self.labels, self.train = paths, labels, train
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return read_clip(self.paths[i], self.train), torch.tensor(self.labels[i], dtype=torch.long)


# %% Cell 10
# Giu dong bo voi src/models/cnn_lstm.py: cung ten tham so (cnn., motion_cnn.,
# lstm., attention., head.) de checkpoint nap thang vao web (src/infer.py).
from torchvision.models import resnet18, ResNet18_Weights

class TemporalAttention(nn.Module):
    def __init__(self, d):
        super().__init__(); self.score = nn.Sequential(nn.Linear(d,d//2), nn.Tanh(), nn.Linear(d//2,1))
    def forward(self, h):
        a = torch.softmax(self.score(h).squeeze(-1), dim=1)
        return torch.bmm(a.unsqueeze(1), h).squeeze(1), a

def _lstm_from_cfg(input_dim, m):
    hidden, layers, bi = m.get("lstm_hidden", 512), m.get("lstm_layers", 2), m.get("bidirectional", True)
    lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True, bidirectional=bi,
                   dropout=m.get("dropout", .5) if layers > 1 else 0.0)
    return lstm, hidden * (2 if bi else 1)

def _pool(z, attention):
    # Khong attention: trong so deu cho moi frame (mean pooling).
    if attention is None:
        return z.mean(dim=1), None
    return attention(z)

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

def spatial_backbone(name):
    """CNN tung frame + do dai vector dac trung (giong src/models/cnn_lstm.py)."""
    if name == "mobilenet_v2":
        net = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        return nn.Sequential(net.features, nn.AdaptiveAvgPool2d(1)), 1280
    net = resnet50(weights=ResNet50_Weights.DEFAULT)
    return nn.Sequential(*list(net.children())[:-1]), 2048

class ResNet50BiLSTMAttention(nn.Module):
    """Ten giu nguyen cho checkpoint cu; backbone chon bang m["backbone"] (resnet50 / mobilenet_v2)."""
    def __init__(self, m):
        super().__init__()
        self.cnn, self.feat_dim = spatial_backbone(m.get("backbone", "resnet50"))
        # temporal="none": chi dac trung khong gian, trung binh 2048 chieu qua cac frame.
        self.lstm, out_dim = (None, self.feat_dim) if m.get("temporal", "lstm") == "none" else _lstm_from_cfg(self.feat_dim, m)
        self.attention = TemporalAttention(out_dim) if m.get("use_attention", True) else None
        self.head = nn.Sequential(nn.Dropout(m.get("dropout", .5)), nn.Linear(out_dim, 2))
    def forward(self, x):
        b,t,c,h,w = x.shape; z = self.cnn(x.reshape(b*t,c,h,w)).flatten(1)
        z = z.view(b,t,self.feat_dim)
        if self.lstm is not None: z,_ = self.lstm(z)
        context, alpha = _pool(z, self.attention)
        return self.head(context), alpha

def motion_frames(x):
    """|frame_t - frame_{t-1}| in pixel space from a normalised RGB clip, frame 0 black."""
    mean = x.new_tensor(MEAN).view(1,1,3,1,1); std = x.new_tensor(STD).view(1,1,3,1,1)
    raw = x * std + mean
    diff = torch.cat([torch.zeros_like(raw[:, :1]), (raw[:, 1:] - raw[:, :-1]).abs()], dim=1)
    return (diff - mean) / std

class TwoStreamBiLSTMAttention(nn.Module):
    """Spatial stream (ResNet50, RGB) + motion stream (ResNet18, frame diff), fused per frame."""
    def __init__(self, m):
        super().__init__()
        self.cnn = nn.Sequential(*list(resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-1])
        self.motion_cnn = nn.Sequential(*list(resnet18(weights=ResNet18_Weights.DEFAULT).children())[:-1])
        self.lstm, out_dim = _lstm_from_cfg(2048 + 512, m)
        self.attention = TemporalAttention(out_dim) if m.get("use_attention", True) else None
        self.head = nn.Sequential(nn.Dropout(m.get("dropout", .5)), nn.Linear(out_dim, 2))
    def forward(self, x):
        b,t,c,h,w = x.shape
        appearance = self.cnn(x.reshape(b*t,c,h,w)).flatten(1).view(b,t,2048)
        moving = self.motion_cnn(motion_frames(x).reshape(b*t,c,h,w)).flatten(1).view(b,t,512)
        z,_ = self.lstm(torch.cat([appearance, moving], dim=2)); context, alpha = _pool(z, self.attention)
        return self.head(context), alpha

MODEL_CFG = {"arch": EXP["arch"], "temporal": EXP.get("temporal", "lstm"), "backbone": EXP.get("backbone", "resnet50"), "pretrained": False, "freeze_backbone": False,
             "lstm_hidden": 512, "lstm_layers": 2, "bidirectional": EXP["bidirectional"], "dropout": .5,
             "use_attention": EXP["attention"], "num_classes": 2}

def build_model():
    cls = TwoStreamBiLSTMAttention if EXP["arch"] == "two_stream" else ResNet50BiLSTMAttention
    return cls(MODEL_CFG)


# %% Cell 12
def load_old(model, path):
    """Nap tung khoi (cnn, motion_cnn, lstm, attention, head) khi MOI tensor cua khoi
    khop ten va kich thuoc; tra ve ten cac tham so da nap.

    Bien the khac kien truc (LSTM 1 chieu, 2 luong...) chi nhan khoi khop, vd
    ResNet50 cua luong RGB. Neu LSTM khong nap duoc thi attention/head cung khoi
    tao moi: chung da hoc tren dau ra cua LSTM cu.
    """
    if not path or INIT_FROM == "imagenet":
        print("Khoi tao tu ImageNet (khong nap checkpoint).")
        return model, set()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt); mapped = {}
    # Supports project checkpoints (spatial/temporal) and this notebook format.
    for k,v in state.items():
        k = k.replace("spatial.features.", "cnn.").replace("temporal.lstm.", "lstm.")
        k = k.replace("temporal.attn.", "attention.").replace("temporal.classifier.", "head.")
        mapped[k] = v
    own = model.state_dict()
    blocks = {k.split(".")[0] for k in own}
    def block_ok(b):
        keys = [k for k in own if k.split(".")[0] == b]
        return all(k in mapped and mapped[k].shape == own[k].shape for k in keys)
    ok = {b for b in blocks if block_ok(b)}
    if "lstm" not in ok:
        ok -= {"attention", "head"}
    usable = {k: mapped[k] for k in own if k.split(".")[0] in ok}
    model.load_state_dict(usable, strict=False)
    print(f"Loaded checkpoint: {path} | khoi da nap: {sorted(ok)} | khoi moi: {sorted(blocks - ok)}")
    return model, set(usable)


# %% Cell 13
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve, average_precision_score

def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None; model.train(train); ys=[]; probs=[]; losses=[]
    # Keep CNN BatchNorm statistics fixed during webcam adaptation.
    if train and getattr(model, "freeze_cnn_stats", False):
        model.cnn.eval()
        if hasattr(model, "motion_cnn"): model.motion_cnn.eval()
    with (torch.enable_grad() if train else torch.no_grad()):
        for x,y in tqdm(loader, leave=False):
            x,y = x.to(DEVICE), y.to(DEVICE)
            if train: optimizer.zero_grad(set_to_none=True)
            logits,_ = model(x); loss = criterion(logits,y)
            if train: loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            losses.append(loss.item()); ys += y.cpu().tolist()
            probs += torch.softmax(logits.detach().float(), 1)[:, 1].cpu().tolist()
    return {"loss": float(np.mean(losses)), "y": ys, "p": probs}

def metrics(y, p, thr=0.5):
    """Chi so cho bao cao. Lop duong = violence."""
    y = np.asarray(y); pred = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {"threshold": float(thr), "n": int(len(y)),
           "accuracy": accuracy_score(y, pred),
           "precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred, zero_division=0),
           "f1": f1_score(y, pred, zero_division=0),
           "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
           "false_alarm_rate": fp / (fp + tn) if fp + tn else 0.0,
           "roc_auc": roc_auc_score(y, p) if len(set(y.tolist())) == 2 else float("nan"),
           "pr_auc": average_precision_score(y, p) if len(set(y.tolist())) == 2 else float("nan"),
           "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}

def pick_threshold(y, p):
    """Nguong toi da macro-F1 tren VALIDATION. Khong bao gio chon tren test."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    return float(max(grid, key=lambda t: (metrics(y, p, t)["f1_macro"], -abs(t - 0.5))))

def short(m):
    return {k: m[k] for k in ("loss", "accuracy", "f1_macro", "roc_auc") if k in m}


# %% Cell 14
# ===== BASE TRAINING -> WEBCAM FINE-TUNING =====
if DATA_MODE != "combined" or TRAIN_SOURCES is not None:
    raise ValueError("Two-stage training requires combined mode and TRAIN_SOURCES=None.")
if FINETUNE_EPOCHS <= FREEZE_EPOCHS or FREEZE_EPOCHS < 1:
    raise ValueError("Require 1 <= FREEZE_EPOCHS < FINETUNE_EPOCHS.")
if REPLAY_RATIO < 0:
    raise ValueError("REPLAY_RATIO must be nonnegative.")
loader = lambda p,y,tr=False: DataLoader(VideoDataset(p,y,tr), batch_size=BATCH,
    shuffle=tr, num_workers=0, pin_memory=DEVICE.type=="cuda")

def rows_for(sources, split):
    df = manifest[manifest.source.isin(sources) & manifest.split.eq(split)].copy()
    if df.empty or set(df.label) != {0, 1}:
        raise ValueError(f"{sources} / {split}: need both classes. Check dataset Input and manifest.")
    return df

def xy(df): return df.resolved_path.tolist(), df.label.astype(int).tolist()

# Validate all splits before spending time on training.
public = {s: rows_for(BASE_SOURCES, s) for s in ("train", "val", "test")}
webcam = {s: rows_for(WEBCAM_SOURCES, s) for s in ("train", "val", "test")}
train_p, train_y = xy(webcam["train"])
val_p, val_y = xy(webcam["val"])
test_p, test_y = xy(webcam["test"])
val_loader, test_loader = loader(val_p,val_y), loader(test_p,test_y)
criterion = nn.CrossEntropyLoss()
model = build_model().to(DEVICE)

def checkpoint_cfg(threshold):
    return {"data": {"classes": ["non_violence", "violence"], "num_frames": T,
                     "image_size": SIZE, "frame_diff": EXP["input"] == "diff"},
            "model": MODEL_CFG,
            "infer": {"window_frames": T, "stride": max(1,T//2), "threshold": threshold},
            "experiment": EXPERIMENT}

def configure_finetune(unfreeze):
    for p in model.parameters(): p.requires_grad_(True)
    backbones = [model.cnn] + ([model.motion_cnn] if hasattr(model,"motion_cnn") else [])
    for backbone in backbones:
        for p in backbone.parameters(): p.requires_grad_(False)
        if unfreeze:
            block = backbone[0][-1] if MODEL_CFG["backbone"] == "mobilenet_v2" else backbone[-2]
            for p in block.parameters(): p.requires_grad_(True)
    model.freeze_cnn_stats = True
    cnn_ids = {id(p) for b in backbones for p in b.parameters()}
    groups = [dict(params=[p for p in model.parameters() if p.requires_grad and id(p) not in cnn_ids], lr=FINETUNE_HEAD_LR)]
    cnn = [p for p in model.parameters() if p.requires_grad and id(p) in cnn_ids]
    if cnn: groups.append(dict(params=cnn, lr=FINETUNE_CNN_LR))
    return torch.optim.AdamW(groups, weight_decay=1e-4)

def fit_stage(name, train_df, val_df, epochs, patience, finetune=False):
    train_loader = loader(*xy(train_df), tr=True)
    validation_loader = loader(*xy(val_df))
    optimizer = configure_finetune(False) if finetune else torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-4)
    path = RUN_DIR / ("best.pt" if finetune else "base_best.pt")
    best, stale, history = -1.0, 0, []
    for epoch in range(1,epochs+1):
        if finetune and epoch == FREEZE_EPOCHS + 1:
            optimizer = configure_finetune(True)
            stale = 0
            print("Unfreeze final CNN block; keep BatchNorm statistics fixed.")
        tr = run_epoch(model,train_loader,criterion,optimizer)
        va = run_epoch(model,validation_loader,criterion)
        tr_m = {"loss": tr["loss"], **metrics(tr["y"],tr["p"])}
        va_m = {"loss": va["loss"], **metrics(va["y"],va["p"])}
        history.append({"epoch":epoch, **{f"train_{k}":v for k,v in short(tr_m).items()},
                        **{f"val_{k}":v for k,v in short(va_m).items()}})
        pd.DataFrame(history).to_csv(RUN_DIR / f"{name}_history.csv",index=False)
        print(f"{name} epoch {epoch:02d} train={short(tr_m)} val={short(va_m)}")
        if va_m["f1_macro"] > best:
            best, stale = va_m["f1_macro"], 0
            torch.save({"model":model.state_dict(), "cfg":checkpoint_cfg(.5),
                        "val":va_m, "stage":name, "epoch":epoch},path)
        else:
            stale += 1
            if stale >= patience and (not finetune or epoch > FREEZE_EPOCHS):
                print(f"Early stopping: {patience} epochs without validation improvement.")
                break
    model.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=False)["model"],strict=True)
    return path

if BASE_CHECKPOINT:
    base_path = Path(BASE_CHECKPOINT)
    base_ckpt = torch.load(base_path,map_location=DEVICE,weights_only=False)
    if base_ckpt.get("cfg",{}).get("model") != MODEL_CFG:
        raise ValueError("BASE_CHECKPOINT architecture/config must match MODEL_CFG exactly.")
    model.load_state_dict(base_ckpt["model"],strict=True)
else:
    if INIT_FROM != "imagenet":
        raise ValueError("Use INIT_FROM='imagenet', or supply BASE_CHECKPOINT for strict loading.")
    base_path = fit_stage("base",public["train"],public["val"],BASE_EPOCHS,PATIENCE)

# Compare base and adapted model on webcam validation, never choose on test.
base_va = run_epoch(model,val_loader,criterion)
base_thr = pick_threshold(base_va["y"],base_va["p"])
base_validation = metrics(base_va["y"],base_va["p"],base_thr)
print("Base model on webcam validation:",base_validation)

adapt_train = webcam["train"].copy()
if REPLAY_RATIO:
    count = min(len(public["train"]), round(len(adapt_train)*REPLAY_RATIO))
    if count:
        replay = public["train"].sample(n=count,random_state=TRAIN_SEED)
        adapt_train = pd.concat([adapt_train,replay],ignore_index=True)
train_loader = loader(*xy(adapt_train),tr=True)
best_path = fit_stage("finetune",adapt_train,webcam["val"],FINETUNE_EPOCHS,FINETUNE_PATIENCE,True)
print("BASE:",base_path,"| FINETUNED:",best_path)


# %% Cell 15
# ===== DANH GIA: nguong tu validation -> test, theo nguon, do tre =====
import json, time
import matplotlib.pyplot as plt

ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model"]); model.eval()

va = run_epoch(model, val_loader, criterion)
thr = pick_threshold(va["y"], va["p"])
te = run_epoch(model, test_loader, criterion)
result = {"experiment": EXPERIMENT, "seed": TRAIN_SEED, "config": EXP, "init_from": INIT_FROM,
          "train_sources": sorted(WEBCAM_SOURCES),
          "training_strategy": "public_base_then_webcam_finetune",
          "base_checkpoint": str(base_path), "replay_ratio": REPLAY_RATIO,
          "base_webcam_validation": base_validation,
          "val_at_threshold": metrics(va["y"], va["p"], thr),
          "test_at_0.5": metrics(te["y"], te["p"], 0.5),
          "test": metrics(te["y"], te["p"], thr)}
print(f"Nguong chon tren validation: {thr}")
print("TEST @thr:", result["test"])

# Ket qua theo tung dataset nguon (cross-dataset khi TRAIN_SOURCES != None).
if "source" in manifest.columns:
    src_of = dict(zip(manifest.resolved_path, manifest.source))
    sources = np.array([src_of[p] for p in test_p])
    per_source = {}
    for s in sorted(set(sources)):
        idx = np.where(sources == s)[0]
        ys, ps = [te["y"][i] for i in idx], [te["p"][i] for i in idx]
        per_source[s] = {"role": "cross" if TRAIN_SOURCES and s not in TRAIN_SOURCES else "in-domain",
                         **metrics(ys, ps, thr)}
    result["test_per_source"] = per_source
    print(pd.DataFrame(per_source).T[["role", "n", "accuracy", "precision", "recall", "f1", "roc_auc"]].to_string())

# Do tre 1 clip (T frame): GPU Kaggle va CPU (gan voi may demo localhost).
def latency_ms(m, device, runs=5):
    m = m.to(device).eval(); x = torch.randn(1, T, 3, SIZE, SIZE, device=device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    with torch.no_grad():
        m(x); sync(); start = time.perf_counter()
        for _ in range(runs): m(x)
        sync()
    return round((time.perf_counter() - start) / runs * 1000, 1)
result["latency_ms"] = {"device": latency_ms(model, DEVICE), "cpu": latency_ms(model, torch.device("cpu"), runs=2)}
result["params_million"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
model.to(DEVICE)
print("Latency / clip (ms):", result["latency_ms"], "| params (M):", result["params_million"])

# Confusion matrix + ROC cho bao cao.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
t = result["test"]; cm = np.array([[t["tn"], t["fp"]], [t["fn"], t["tp"]]])
ax1.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm): ax1.text(j, i, v, ha="center", va="center")
ax1.set_xticks([0, 1], ["non_violence", "violence"]); ax1.set_yticks([0, 1], ["non_violence", "violence"])
ax1.set_xlabel("Du doan"); ax1.set_ylabel("Thuc te"); ax1.set_title(f"{EXPERIMENT} @ thr={thr}")
fpr, tpr, _ = roc_curve(te["y"], te["p"])
ax2.plot(fpr, tpr, label=f"AUC = {t['roc_auc']}"); ax2.plot([0, 1], [0, 1], "--", color="gray")
ax2.set_xlabel("False positive rate"); ax2.set_ylabel("True positive rate"); ax2.legend(); ax2.set_title("ROC (test)")
fig.tight_layout(); fig.savefig(RUN_DIR / "confusion_roc.png", dpi=150); plt.show()

# group_id/source cho bootstrap theo nhom va bang theo nguon (src/research/aggregate.py).
meta = manifest.set_index("resolved_path")
def predictions(paths, out):
    return pd.DataFrame({"path": paths, "group_id": meta.loc[paths, "group_id"].values,
                         "source": meta.loc[paths, "source"].values if "source" in meta else "",
                         "label": out["y"], "prob": out["p"], "pred": (np.array(out["p"]) >= thr).astype(int)})
predictions(test_p, te).to_csv(RUN_DIR / "test_predictions.csv", index=False)
predictions(val_p, va).to_csv(RUN_DIR / "val_predictions.csv", index=False)
(RUN_DIR / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
# Ghi nguong da chon vao checkpoint: web (src/infer.py) doc nguong tu day.
ckpt["cfg"] = checkpoint_cfg(thr)
torch.save(ckpt, best_path)
print("Saved:", RUN_DIR / "metrics.json", "|", best_path, "threshold =", thr)


# %% Cell 16
# ===== E5 DO BEN: danh gia lai test khi FPS / JPEG / do sang thay doi =====
# Tat mac dinh vi ton thoi gian (moi dieu kien = 1 lan chay het tap test).
RUN_ROBUSTNESS = False
CONDITIONS = {
    "goc": {},
    "fps_15": {"fps_step": 2}, "fps_10": {"fps_step": 3}, "fps_5": {"fps_step": 6},
    "jpeg_90": {"jpeg": 90}, "jpeg_70": {"jpeg": 70}, "jpeg_50": {"jpeg": 50},
    "toi_70": {"brightness": 0.7}, "sang_130": {"brightness": 1.3},
}
if RUN_ROBUSTNESS:
    robustness = {}
    try:
        for name, cond in CONDITIONS.items():
            PERTURB = cond
            out = run_epoch(model, test_loader, criterion)
            robustness[name] = {"perturb": cond, **metrics(out["y"], out["p"], thr)}
            print(name, short({"loss": out["loss"], **robustness[name]}))
    finally:
        PERTURB = {}
    (RUN_DIR / "robustness.json").write_text(json.dumps(robustness, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved:", RUN_DIR / "robustness.json")


# %% Cell 17
# ===== BANG ABLATION: gop moi OUT_DIR/<EXPERIMENT>/metrics.json da chay =====
# Kaggle xoa /kaggle/working giua cac phien: tai metrics.json cua moi lan chay ve
# (hoac Add Output lam Input) roi dat vao OUT_DIR/<ten>/ truoc khi chay cell nay.
rows = []
for path in sorted(OUT_DIR.glob("*/metrics.json")):
    r = json.loads(path.read_text(encoding="utf-8")); t = r["test"]
    rows.append({"experiment": r["experiment"], "seed": r.get("seed", 42), "input": r["config"]["input"], "arch": r["config"]["arch"],
                 "bilstm": r["config"]["bidirectional"], "attention": r["config"]["attention"],
                 "frames": r["config"]["frames"], "init": r["init_from"], "train_sources": r["train_sources"],
                 "threshold": t["threshold"], "accuracy": t["accuracy"], "precision": t["precision"],
                 "recall": t["recall"], "f1": t["f1"], "roc_auc": t["roc_auc"],
                 "false_alarm": t["false_alarm_rate"], "gpu_ms": r["latency_ms"]["device"],
                 "cpu_ms": r["latency_ms"]["cpu"], "params_M": r["params_million"]})
ablation = pd.DataFrame(rows)
if len(ablation):
    ablation.to_csv(OUT_DIR / "ablation.csv", index=False)
    print(ablation.to_string(index=False))
    cols = ["accuracy", "f1", "roc_auc", "false_alarm"]
    summary = ablation.groupby(["experiment", "init", "train_sources"])[cols].agg(["mean", "std", "count"]).round(4)
    print("\nMean +- std qua cac seed:\n" + summary.to_string())
    print("Kiem dinh thong ke (bootstrap CI, McNemar): python -m src.research.aggregate <thu muc ket qua>")
    # So sanh chi co y nghia giua cac dong cung init va cung train_sources.
    if ablation[["init", "train_sources"]].astype(str).drop_duplicates().shape[0] > 1:
        print("\nCANH BAO: cac dong khac INIT_FROM/TRAIN_SOURCES, khong so sanh truc tiep.")

# So lieu da cong bo, CHI de tham chieu: dataset/cach chia khac manifest gop o tren.
# Chi so truc tiep khi cung dataset va split; combined khong so truc tiep voi RWF-2000.
published = pd.DataFrame([
    {"method": "ConvLSTM + frame diff (Sudhakaran & Lanz, AVSS 2017)", "dataset": "Hockey / Movies / Violent-Flows", "accuracy": "xem bai bao"},
    {"method": "VGG16 + LSTM (Soliman et al., ICICIS 2019)", "dataset": "RLVS", "accuracy": 0.882},
    {"method": "Flow Gated Network (Cheng et al., ICPR 2020)", "dataset": "RWF-2000 (split chinh thuc)", "accuracy": 0.8725},
])
print("\nTham chieu cong bo:\n" + published.to_string(index=False))

