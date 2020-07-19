import torch
import numpy as np
from mannequinchallenge.options.train_options import TrainOptions
from mannequinchallenge.loaders import aligned_data_loader
from mannequinchallenge.models import pix2pix_model

def infer_depth(img):
    BATCH_SIZE = 1

    opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

    eval_num_threads = 2
    video_data_loader = aligned_data_loader.PLARRDataLoader(img, BATCH_SIZE)
    video_dataset = video_data_loader.load_data()

    model = pix2pix_model.Pix2PixModel(opt)

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    best_epoch = 0
    global_step = 0

    print(
        '=================================  BEGIN VALIDATION ====================================='
    )

    model.switch_to_eval()

    for i, data in enumerate(video_dataset):
        print(i)
        stacked_img = data[0]
        disp_img =  model.run_PLARR(stacked_img)
        disp_img = disp_img.resize(img.size)
        disp_array = np.array(disp_img)
        return disp_array, disp_img