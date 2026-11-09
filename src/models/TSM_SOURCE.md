# Local TSM trial

`checkpoints/tsm_resnet50.pt` is a copy of the user-supplied
`MobileNet-TSM/checkpoint.pt`. Despite the folder name, its tensors match
ResNet50 with temporal shifts and SE attention in layer3/layer4.
All parameters are loaded strictly; no unmatched tensors are discarded.

The adapter follows the supplied temporal_shift.py and SEAttention.py.
SE runs after bn3 and before residual addition, as shown in the commented
SE path in MODEL/eca_resnet1.py. The supplied source currently comments
out that path, so exact training-time behavior still needs confirmation.
The checkpoint contains only a state dict, with no training configuration.
Eight RGB segments, shift division 8, mean consensus and ImageNet
normalization follow the source defaults. The accompanying class_indices.json
defines 0=violence and 1=Noviolence; the adapter reverses outputs to match
the application's 0=non_violence and 1=violence convention.

configs/webcam_trial.json selects this model for webcam and upload.
Video links reuse the upload route. Threshold 0.5 is a trial setting,
not a validated operating threshold. TSM uses full frames without the
CNN-LSTM motion suppression or MoViNet veto. YOLO only supplies overlays.
To restore the prior trial, set both checkpoint fields to final_best.pt.

Smoke verification: strict loading and a forward pass succeed; synthetic
8-frame inputs return HTTP 200 from /predict/live, /predict and /predict/url.
The URL test substitutes a local video for the network download; alerts
and annotation are stubbed to avoid sending messages during verification.
Real camera access, network downloads and recognition quality require
testing with user footage. The temporary test database cleanup encountered
a Windows file lock after all three route assertions passed.
