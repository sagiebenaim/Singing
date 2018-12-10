from utils import get_config
from trainer import MUSIC_Trainer
from test import GenerateMusic, music_track
from tqdm import tqdm
import numpy as np
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import torch
import os
import librosa
import dsdtools
import soundfile as sf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='DSD100', help='evaluating DSD100 or customize directory')
opts = parser.parse_args()

def create_songs_dsd(config, checkpoint,  output_folder='./data/dsd_outputs/',
                 method_name='method_1', target='vocals'):
    """
    Creating vocals and accompaniment tracks in DSD100 format for later evaluation in matlab official code
    :param config: path to config file
    :param checkpoint_dir: path to generator's saved parameters
    :param output_folder: desired output path
    :param method_name: name of method
    :param target: desired target (vocals/drums/bass)
    """
    dsd = dsdtools.DB(root_dir='../../data/datasets/music/DSD100')
    tracks = dsd.load_dsd_tracks()

    output_folder = os.path.join(output_folder, method_name)
    test_folder = os.path.join(output_folder, 'Test')
    dev_folder = os.path.join(output_folder, 'Dev')
    if not os.path.exists(output_folder):
        print("Creating directory: {}".format(output_folder))
        print("Creating directory: {}".format(test_folder))
        print("Creating directory: {}".format(dev_folder))
        os.makedirs(output_folder), os.makedirs(test_folder), os.makedirs(dev_folder)

    config = get_config(config)
    trainer = MUSIC_Trainer(config)
    last_gen_name = checkpoint
    state_dict = torch.load(last_gen_name)
    trainer.gen.load_state_dict(state_dict['gen'])
    trainer.cuda()
    trainer.eval()
    encode, decode = trainer.gen.encode, trainer.gen.decode
    enhance = 9

    print(method_name)
    for track in tqdm(tracks):

        sample_rate = 20480
        music_array, music_array_ref, music_array_inter = music_track(track, target)
        music_array = librosa.resample(music_array.transpose(), track.rate, sample_rate)

        masker_l = GenerateMusic(music_array[0, :], encode, decode, enhance=enhance)
        recon_vocals_l, recon_inter_l = masker_l.forward()
        masker_r = GenerateMusic(music_array[1, :], encode, decode, enhance=enhance)
        recon_vocals_r, recon_inter_r = masker_r.forward()

        recon_vocals = np.vstack((recon_vocals_l, recon_vocals_r))
        recon_inter = np.vstack((recon_inter_l, recon_inter_r))
        recon_vocals = librosa.resample(recon_vocals, sample_rate, track.rate)
        recon_inter = librosa.resample(recon_inter, sample_rate, track.rate)

        recon_vocals = recon_vocals.transpose()
        recon_inter = recon_inter.transpose()

        if len(music_array_ref) > len(recon_vocals):
            len_diff = len(music_array_ref) - len(recon_vocals)
            recon_vocals = np.concatenate((recon_vocals, recon_vocals[-len_diff:, :]))
            recon_inter = np.concatenate((recon_inter, recon_inter[-len_diff:, :]))
        elif len(music_array_ref) < len(recon_vocals):
            recon_vocals = recon_vocals[0:len(music_array_ref), :]
            recon_inter = recon_inter[0:len(music_array_ref), :]

        track_dir_name = track.filename
        if track.subset == 'Test':
            save_dir = os.path.join(test_folder, track_dir_name)
        else:
            save_dir = os.path.join(dev_folder, track_dir_name)

        if not os.path.exists(save_dir):
            print("Creating directory: {}".format(save_dir))
            os.makedirs(save_dir)
        sf.write(save_dir + '/vocals.wav', recon_vocals, track.rate)
        sf.write(save_dir + '/accompaniment.wav', recon_inter, track.rate)


def create_songs(config, checkpoint,  output_folder='./outputs',
                 input_folder='./inputs'):
    """
    creating vocals and accompaniment for any given song directory, for format supported by soundfile library
    :param config: path to .yaml file containing hyperparameters for the trainer
    :param checkpoint_dir: path to the generator checkpoint
    :param output_folder:
    :param input_folder:
    """
    if not os.path.exists(output_folder):
        print("Creating directory: {}".format(output_folder))
        os.makedirs(output_folder)

    config = get_config(config)
    trainer = MUSIC_Trainer(config)
    last_gen_name = checkpoint
    state_dict = torch.load(last_gen_name)
    trainer.gen.load_state_dict(state_dict['gen'])
    trainer.cuda()
    trainer.eval()
    encode = trainer.gen.encode  # encode function
    decode = trainer.gen.decode  # decode function

    music_mix_list = os.listdir(input_folder)
    enhance = 9
    for musicfile_mix in tqdm(music_mix_list):

        sample_rate = 20480
        music_array, old_sr = sf.read(input_folder + '/' + musicfile_mix)
        music_array = librosa.resample(music_array, old_sr, sample_rate)

        if len(music_array.shape) > 1:
            masker_l = GenerateMusic(music_array[0, :], encode, decode, enhance=enhance)
            recon_vocals_l, recon_inter_l = masker_l.forward()
            masker_r = GenerateMusic(music_array[1, :], encode, decode, enhance=enhance)
            recon_vocals_r, recon_inter_r = masker_r.forward()
        else:
            masker_l = GenerateMusic(music_array, encode, decode, enhance=enhance)
            recon_vocals_l, recon_inter_l = masker_l.forward()
            recon_vocals_r, recon_inter_r = recon_vocals_l, recon_inter_l

        recon_vocals = np.vstack((recon_vocals_l, recon_vocals_r))
        recon_inter = np.vstack((recon_inter_l, recon_inter_r))
        recon_vocals = librosa.resample(recon_vocals, sample_rate, old_sr)
        recon_inter = librosa.resample(recon_inter, sample_rate, old_sr)

        sf.write(output_folder + 'vocals_' + musicfile_mix, recon_vocals.transpose(), old_sr)
        sf.write(output_folder + 'accompaniment_' + musicfile_mix, recon_inter.transpose(), old_sr)


if __name__ == '__main__':

    if opts.input == 'DSD100':
        create_songs_dsd(config='./configs/vocals_new.yaml',
                     checkpoint='./models/singing_model.pt',  output_folder='./outputs/',
                     method_name='method_1', target='vocals')
    elif opts.input == 'custom':
        create_songs(config='./configs/vocals_new.yaml', checkpoint='./models/singing_model.pt',
                     output_folder='./outputs/custom', input_folder='./inputs')