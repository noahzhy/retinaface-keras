import keras
from nets.retinaface import RetinaFace

if __name__ == "__main__":
    cfg = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'image_size': 640,
    'in_channel': 32,
    'out_channel': 64
    }

    model = RetinaFace(cfg, backbone="mobilenet")
    model.summary()