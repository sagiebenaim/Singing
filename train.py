"""
modified from: Multimodal Unsupervised Image-to-Image Translation
https://github.com/NVlabs/MUNIT
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images
import argparse
from torch.autograd import Variable
from trainer import MUSIC_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil
from tqdm import tqdm
from test import test_stft_dsd

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/vocals_new.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='./data/singing_outputs', help="outputs path")
parser.add_argument('--resume', default=0, action="store_true")
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path
config_name = opts.config.split('/')[1]
config_name = ', ' + config_name.split('.')[0]

# Setup model and data loader
trainer = MUSIC_Trainer(config)
trainer.cuda()
train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)
train_display_images_a = Variable(torch.stack([train_loader_a.dataset[i] for i in range(display_size)]).cuda())
train_display_images_b = Variable(torch.stack([train_loader_b.dataset[i] for i in range(display_size)]).cuda())
test_display_images_a = Variable(torch.stack([test_loader_a.dataset[i] for i in range(display_size)]).cuda())
test_display_images_b = Variable(torch.stack([test_loader_b.dataset[i] for i in range(display_size)]).cuda())

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory, tests_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder
# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
while True:
    for it, (images_a, images_b) in tqdm(enumerate(zip(train_loader_a, train_loader_b))):
        trainer.update_learning_rate()
        images_a, images_b = Variable(images_a.cuda()), Variable(images_b.cuda())

        # Main training code
        trainer.dis_update(images_a, images_b, config)
        trainer.gen_update(images_a, images_b, config)

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            write_loss(iterations, trainer, train_writer)

        # print result
        if (iterations + 1) % config['print_iter'] == 0:
            print("Music2Vocals STFT Iteration: %08d/%08d" % (iterations + 1, max_iter) + config_name)

        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
                train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
            write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
            # HTML
            write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():
                image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(image_outputs, display_size, image_directory, 'train_current')

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        # Save testing music
        if (iterations + 1) % config['snapshot_save_test'] == 0:
            test_output_dir = tests_directory + '/b2a_%08d/' % (iterations + 1)
            # if not os.path.exists(test_output_dir):
            #     os.makedirs(test_output_dir)
            sdr_median, sir_median, sar_median, sdr_max, sir_max = test_stft_dsd(config=opts.config,
                checkpoint_dir=checkpoint_directory, method_name='b2a_%08d' % (iterations + 1),
                output_folder=test_output_dir, target='vocals')
            train_writer.add_scalar('sdr_median', sdr_median, iterations + 1)
            train_writer.add_scalar('sir_median', sir_median, iterations + 1)
            train_writer.add_scalar('sar_median', sar_median, iterations + 1)
            train_writer.add_scalar('sdr_max', sdr_max, iterations + 1)
            train_writer.add_scalar('sir_max', sir_max, iterations + 1)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')

