from pathlib import Path
import random
import numpy as np
import torch

# =========================
# PATH
# =========================

AIRTLAB_ROOT = Path(
    "/kaggle/input/datasets/bohong111/webcammmmm/A-Dataset-for-Automatic-Violence-Detection-in-Videos-master/violence-detection-dataset"
)

def find_rlvs_root():
    preferred = Path("/kaggle/input/datasets/mohamedmustafa/real-life-violence-situations-dataset/Real Life Violence Dataset")
    if (preferred / "NonViolence").is_dir() and (preferred / "Violence").is_dir():
        return preferred
    for folder in sorted(Path("/kaggle/input/datasets/mohamedmustafa/real-life-violence-situations-dataset").rglob("NonViolence")):
        if folder.is_dir() and (folder.parent / "Violence").is_dir():
            return folder.parent
    raise FileNotFoundError("Add Real Life Violence Situations Dataset: need NonViolence/ and Violence/.")

RLVS_ROOT = find_rlvs_root()
RWF_ROOT = Path("/kaggle/input/datasets/vulamnguyen/rwf2000/RWF-2000")

# Merged (messaoudiriham/merged-violence-dataset, 7000 video) KHONG dung nua:
# lon nhat, trung nhieu voi RLVS, va lan at clip webcam tu quay (MyWebcam).

def find_datasetwebcam_root():
    # Dataset Kaggle ten DatasetWebCam. Ben trong la AIRTLab:
    # violence-detection-dataset/non-violent|violent/cam1|cam2
    preferred = AIRTLAB_ROOT
    if (preferred / "non-violent").is_dir() and (preferred / "violent").is_dir():
        return preferred
    named = []
    for path in Path("/kaggle/input").rglob("*"):
        if path.is_dir() and path.name.lower().replace("_", "").replace("-", "") in {"datasetwebcam", "datawebcam", "datawebcamv1"}:
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
        "/kaggle/input/datasets/bohong111/webcammmmm/A-Dataset-for-Automatic-Violence-Detection-in-Videos-master/violence-detection-dataset"
    )
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Khong tim thay DatasetWebCam. Hay Add dataset do vao notebook.")

DATASET_WEBCAM_ROOT = find_datasetwebcam_root()

# Dataset liveweb/DataWebCam đã được loại khỏi lần chạy này.
# Giữ giá trị rỗng để các cell sau kiểm tra MYWEBCAM_ROOT.exists() vẫn chạy.
MYWEBCAM_ROOT = Path("/kaggle/working/disabled_mywebcam_dataset")


if not AIRTLAB_ROOT.exists():
    print("AIRTLab cu khong co trong Input. Dung DatasetWebCam:", DATASET_WEBCAM_ROOT)
    AIRTLAB_ROOT = DATASET_WEBCAM_ROOT

# Not used: the model needs 16 consecutive frames of one video.
#   karandeep98/real-life-violence-and-nonviolence-data : loose .jpg frames, not grouped by video
#   quangkhanh205/airtlab-datasets                      : AIRTLab again as extracted frames (duplicate)

# Exact CNN-BiLSTM checkpoint path, if available. Never use YOLO weights here.
INIT_CHECKPOINT_OVERRIDE = None

def find_init_checkpoint():
    def is_detector_weight(path):
        return any("yolo" in part.lower() for part in path.parts)

    if INIT_CHECKPOINT_OVERRIDE:
        path = Path(INIT_CHECKPOINT_OVERRIDE)
        if not path.is_file():
            raise FileNotFoundError(path)
        if is_detector_weight(path):
            raise ValueError("INIT_CHECKPOINT must be CNN-BiLSTM weights, not YOLO weights.")
        return path
    candidates = set()
    model_roots = (
        Path("/kaggle/input/models/bohong111/modelv0/pytorch/default"),
    )
    for folder in model_roots:
        if folder.is_dir():
            candidates.update(p for p in folder.rglob("*.pt")
                              if p.is_file() and not is_detector_weight(p))
    if not candidates:
        print("No CNN-BiLSTM checkpoint found. Use INIT_FROM='imagenet'; YOLO is separate.")
        return None
    if len(candidates) > 1:
        print("Multiple CNN-BiLSTM checkpoints; set INIT_CHECKPOINT_OVERRIDE:")
        for path in sorted(candidates):
            print("  ", path)
        return None
    return next(iter(candidates))


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
# Upload the exported reviewed-label CSV as Kaggle Input, then set its exact path.
REVIEWED_LABELS_CSV = None

# =========================
# CONFIG
# =========================

# 'combined' gồm AIRTLab / DatasetWebCam + RLVS + RWF2000; không dùng liveweb.
DATA_MODE = "combined"
EVAL_DOMAIN = "external"  # đánh giá trên dữ liệu công khai; không dùng liveweb
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
    "combined": [("AIRTLab", AIRTLAB_ROOT), ("RLVS", RLVS_ROOT), ("RWF2000", RWF_ROOT), ("DatasetWebCam", DATASET_WEBCAM_ROOT)],
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
