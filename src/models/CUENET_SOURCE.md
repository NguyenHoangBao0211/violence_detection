Architecture source: https://github.com/damith92/CUENet
File: Enhanced_Uniformer_V2/slowfast/models/uniformerv2_model.py
Local changes: Python logging instead of SlowFast logging; torchvision
StochasticDepth instead of timm DropPath; CPU linspace for meta construction.
All checkpoint weights load strictly. Optimizer tensors remain memory mapped.
Class order assumes the RWF folder order Fight=0, NonFight=1; this checkpoint
does not include a label map. Verify on labelled clips before assessing accuracy.
The webcam uses consecutive frames and does not reproduce the training
spatial cropping pipeline; this trial is not the paper's benchmark evaluation.
