import copy
import torch
import numpy as np
from torchvision.transforms import Compose
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations


_TRANSFORM_DICT = {}


def register_transform(name):
    def decorator(cls):
        _TRANSFORM_DICT[name] = cls
        return cls
    return decorator


def get_transform(cfg):
    if cfg is None or len(cfg) == 0:
        return None
    tfms = []
    for t_dict in cfg:
        t_dict = copy.deepcopy(t_dict)
        cls = _TRANSFORM_DICT[t_dict.pop('type')]
        tfms.append(cls(**t_dict))
    return Compose(tfms)


@register_transform('rotation')
class RandomRotation(object):

    def __init__(self, rot_type):
        super().__init__()
        assert rot_type is None or rot_type in ('x', 'y', 'z', 'so3')
        self.rot_type = rot_type

    def __call__(self, data):
        points = data['point'].unsqueeze(0)
        if self.rot_type in ('x', 'y', 'z'):
            trot = RotateAxisAngle(angle=torch.rand(points.shape[0])*360, axis=self.rot_type.upper(), degrees=True)
        elif self.rot_type == 'so3':
            trot = Rotate(R=random_rotations(points.shape[0]))
        else:
            return data
        
        points = trot.transform_points(points)  # (1, N, 3)
        data['point'] = points[0]
        return data


@register_transform('dropout')
class RandomPointDropout(object):

    def __init__(self, max_dropout_ratio=0.875):
        super().__init__()
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, data):
        point = data['point'].clone()   # (N, 3)
        dropout_ratio =  np.random.random() * self.max_dropout_ratio
        drop_idx = np.where(np.random.random([point.size(0)]) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            point[drop_idx] = point[0].clone()
            data['point'] = point
        return data


@register_transform('scale')
class RandomScalePointCloud(object):

    def __init__(self, scale_low=0.8, scale_high=1.25):
        super().__init__()
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, data):
        scale = np.random.uniform(self.scale_low, self.scale_high)
        data['point'] = data['point'] * scale
        return data


@register_transform('shift')
class RandomShiftPointCloud(object):

    def __init__(self, shift_range=0.1):
        super().__init__()
        self.shift_range = shift_range

    def __call__(self, data):
        shift = torch.FloatTensor(np.random.uniform(-self.shift_range, self.shift_range, size=[1, 3]))
        data['point'] = data['point'] + shift
        return data


@register_transform('jitter')
class JitterPointCloud(object):

    def __init__(self, sigma, clip):
        super().__init__()
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data):
        noise = torch.clamp(torch.randn_like(data['point']) * self.sigma, min=-self.clip, max=+self.clip)
        data['point'] = data['point'] + noise
        return data
