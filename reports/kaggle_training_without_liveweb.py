# ===== BASE TRAINING -> WEBCAM FINE-TUNING =====
if DATA_MODE != "combined" or TRAIN_SOURCES is not None:
    raise ValueError("Two-stage training requires combined mode and TRAIN_SOURCES=None.")
HAS_WEBCAM_DATA = manifest.source.isin(WEBCAM_SOURCES).any()
if HAS_WEBCAM_DATA and (FINETUNE_EPOCHS <= FREEZE_EPOCHS or FREEZE_EPOCHS < 1):
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
webcam = {s: rows_for(WEBCAM_SOURCES, s) for s in ("train", "val", "test")} if HAS_WEBCAM_DATA else None
evaluation = webcam if HAS_WEBCAM_DATA else public
train_p, train_y = xy(evaluation["train"])
val_p, val_y = xy(evaluation["val"])
test_p, test_y = xy(evaluation["test"])
if not HAS_WEBCAM_DATA:
    print("No mywebcam data: train/evaluate public datasets only; skip webcam fine-tuning.")
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
print("Base model validation domain:", "webcam" if HAS_WEBCAM_DATA else "public", base_validation)

if HAS_WEBCAM_DATA:
    adapt_train = webcam["train"].copy()
    if REPLAY_RATIO:
        count = min(len(public["train"]), round(len(adapt_train)*REPLAY_RATIO))
        if count:
            replay = public["train"].sample(n=count,random_state=TRAIN_SEED)
            adapt_train = pd.concat([adapt_train,replay],ignore_index=True)
    train_loader = loader(*xy(adapt_train),tr=True)
    best_path = fit_stage("finetune",adapt_train,webcam["val"],FINETUNE_EPOCHS,FINETUNE_PATIENCE,True)
    print("BASE:",base_path,"| FINETUNED:",best_path)

else:
    best_path = RUN_DIR / "best.pt"
    torch.save({"model": model.state_dict(), "cfg": checkpoint_cfg(base_thr),
                "val": base_validation, "stage": "base_only",
                "source_checkpoint": str(base_path)}, best_path)
    train_loader = loader(*xy(public["train"]), tr=True)
    print("BASE ONLY:", base_path, "| EXPORTED:", best_path)
