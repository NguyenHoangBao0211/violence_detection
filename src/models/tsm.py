"""Binary ResNet50 TSM adapter for the supplied MobileNet-TSM checkpoint."""
from copy import deepcopy

import torch
from torchvision.models import resnet50
from torchvision.models.resnet import Bottleneck


class SEAttention(torch.nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(torch.nn.Linear(channels, hidden, bias=False),
                                      torch.nn.ReLU(inplace=True),
                                      torch.nn.Linear(hidden, channels, bias=False),
                                      torch.nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(self.avg_pool(x).flatten(1))[:, :, None, None]


class SEBottleneck(Bottleneck):
    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.SE(self.bn3(self.conv3(out)))
        return self.relu(out + identity)


class TemporalShift(torch.nn.Module):
    def __init__(self, net, segments=8, division=8):
        super().__init__()
        self.net = net
        self.segments = segments
        self.division = division

    def forward(self, x):
        nt, channels, height, width = x.shape
        x = x.reshape(-1, self.segments, channels, height, width)
        fold = channels // self.division
        shifted = torch.zeros_like(x)
        shifted[:, :-1, :fold] = x[:, 1:, :fold]
        shifted[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]
        shifted[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        return self.net(shifted.reshape(nt, channels, height, width))


class TSMDetector(torch.nn.Module):
    def __init__(self, state, segments=8):
        super().__init__()
        self.segments = segments
        self.base_model = resnet50(weights=None)
        self.base_model.fc = torch.nn.Dropout(0.5)
        self.new_fc = torch.nn.Linear(2048, 2)
        for name, module in list(self.base_model.named_modules()):
            key = f'base_model.{name}.SE.fc.0.weight'
            if key in state:
                hidden, channels = state[key].shape
                module.__class__ = SEBottleneck
                module.SE = SEAttention(channels, hidden)
        for name, module in list(self.base_model.named_modules()):
            if name.endswith('conv1') and f'base_model.{name}.net.weight' in state:
                parent, attr = name.rsplit('.', 1)
                setattr(self.base_model.get_submodule(parent), attr, TemporalShift(module, segments))
        self.load_state_dict(state, strict=True)

    def forward(self, frames):
        if frames.shape[1] != self.segments:
            raise ValueError(f'TSM expects {self.segments} frames per clip')
        logits = self.new_fc(self.base_model(frames.flatten(0, 1)))
        logits = logits.reshape(frames.shape[0], self.segments, 2).mean(dim=1)
        # Source labels: 0=violence, 1=nonviolence. Pipeline expects the reverse.
        return logits[:, [1, 0]], None


def load_tsm(state, cfg, device):
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    out_cfg = deepcopy(cfg or {})
    # The source opts.py defaults to 8 RGB segments and shift division 8.
    segments = out_cfg.get('tsm', {}).get('num_segments', 8)
    model = TSMDetector(state, segments).to(device).eval()
    out_cfg['model_type'] = 'tsm'
    out_cfg.setdefault('data', {}).update(classes=['non_violence', 'violence'],
                                        image_size=224, num_frames=segments, frame_diff=False)
    out_cfg.setdefault('infer', {}).update(window_frames=segments, stride=segments // 2,
                                         threshold=0.5, batch_size=1)
    return model, out_cfg, device
