import torch
from options.train_options import TrainOptions
from loaders import aligned_data_loader
from models import pix2pix_model
import glob

BATCH_SIZE = 1

opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

# Add the names of frames to txt file
video_dir = 'test_data/MOT/'
item_list = glob.glob(video_dir + "*.png")

with open("test_data/test_mot_video_list.txt", "w") as outfile:
    outfile.write("\n".join(sorted(item_list)))

video_list = 'test_data/test_mot_video_list.txt'

eval_num_threads = 2
video_data_loader = aligned_data_loader.DAVISDataLoader(video_list, BATCH_SIZE)
video_dataset = video_data_loader.load_data()
print('========================= Video dataset #images = %d =========' %
      len(video_data_loader))

model = pix2pix_model.Pix2PixModel(opt)

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

best_epoch = 0
global_step = 0

print(
    '=================================  BEGIN VALIDATION ====================================='
)

print('TESTING ON VIDEO')

model.switch_to_eval()
save_path = 'test_data/mot_predictions/'
print('save_path %s' % save_path)

for i, data in enumerate(video_dataset):
    print(i)
    stacked_img = data[0]
    targets = data[1]
    model.run_and_save_DAVIS(stacked_img, targets, save_path)
