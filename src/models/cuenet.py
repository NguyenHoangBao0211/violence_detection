"""UniFormerV2 checkpoint adapter for the CUE-Net webcam trial."""
from copy import deepcopy
import torch
import yaml
from src.data.transforms import ClipTransform
from src.models.uniformerv2_vendor import uniformerv2_l14_336


class CUEDetector(torch.nn.Module):
    def __init__(self, backbone, mean, std):
        super().__init__()
        self.backbone = backbone
        self.clip_transform = ClipTransform(336, False, mean, std)

    def forward(self, frames):
        logits = self.backbone(frames.permute(0, 2, 1, 3, 4).contiguous())
        # RWF directory ordering: Fight=0, NonFight=1.
        return logits[:, [1, 0]], None


def load_cuenet(checkpoint, cfg, device):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    training = yaml.safe_load(saved['cfg'])
    u = training['UNIFORMERV2']
    if u['BACKBONE'] != 'uniformerv2_l14_336':
        raise ValueError('Unsupported CUE-Net backbone')
    with torch.device('meta'):
        backbone = uniformerv2_l14_336(
            pretrained=False, use_checkpoint=False, t_size=training['DATA']['NUM_FRAMES'],
            temporal_downsample=u['TEMPORAL_DOWNSAMPLE'], no_lmhra=u['NO_LMHRA'],
            double_lmhra=u['DOUBLE_LMHRA'], return_list=u['RETURN_LIST'],
            n_layers=u['N_LAYERS'], n_dim=u['N_DIM'], n_head=u['N_HEAD'],
            mlp_factor=u['MLP_FACTOR'], mlp_dropout=u['MLP_DROPOUT'],
            cls_dropout=u['CLS_DROPOUT'], num_classes=training['MODEL']['NUM_CLASSES'],
        )
    backbone.load_state_dict({k.removeprefix('backbone.'): v for k, v in saved['model_state'].items()}, strict=True, assign=True)
    model = CUEDetector(backbone, training['DATA']['MEAN'], training['DATA']['STD']).to(device).eval()
    out = deepcopy(cfg or {})
    out['model_type'] = 'cuenet'
    out.setdefault('data', {}).update(classes=['non_violence', 'violence'], num_frames=64, image_size=336, frame_diff=False)
    out.setdefault('infer', {}).update(window_frames=64, stride=32, threshold=.5, batch_size=1)
    return model, out, device
