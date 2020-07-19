import torch
import numpy as np
from mannequinchallenge.options.train_options import TrainOptions
from mannequinchallenge.loaders import aligned_data_loader
from mannequinchallenge.models import pix2pix_model

model = None

class DictX(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return '<DictX ' + dict.__repr__(self) + '>'

opt = DictX({
    'input': 'single_view',
    "simple_keypoints": 0,
    "mode": "Ours_Bilinear",
    "human_data_term": 0,
    'batchSize': 8,
    'loadSize': 286,
    'fineSize': 256,
    'output_nc': 3,
    'ngf': 64,
    'ndf': 64,
    'which_model_netG': 'unet_256',
    'gpu_ids': '0,1,2,3',
    'name': 'test_local',
    'model': 'pix2pix',
    'nThreads': 2,
    'checkpoints_dir': './monoculardepth/mannequinchallenge/checkpoints/',
    'norm': 'instance',
    'display_winsize': 256,
    'display_id': 1, 
    'identity': 0,
    'max_dataset_size': float("inf"),
    'display_freq': 100,
    'print_freq': 100,
    'save_latest_freq': 5000,
    'save_epoch_freq': 5,
    'phase': 'train',
    'which_epoch': 'latest',
    'niter': 100,
    'niter_decay': 100,
    'lr_decay_epoch': 8,
    'lr_policy': 'step',
    'beta1': 0.5,
    'lr': 0.0004,
    'lambda_A': 10.0,
    'lambda_B': 10.0,
    'pool_size': 50,
    'isTrain': False
})

def infer_depth(img):
    global model
    BATCH_SIZE = 1

    # opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

    video_data_loader = aligned_data_loader.PLARRDataLoader(img, BATCH_SIZE)
    video_dataset = video_data_loader.load_data()

    if model == None:
        model = pix2pix_model.Pix2PixModel(opt)
        model.switch_to_eval()

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    for i, data in enumerate(video_dataset):
        stacked_img = data[0]
        disp_img =  model.run_PLARR(stacked_img)
        disp_img = disp_img.resize(img.size)
        disp_array = np.array(disp_img)
        return disp_array, disp_img