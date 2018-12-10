from utils import get_config, get_model_list
from trainer import MUSIC_Trainer
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import torch
import os
import librosa
import museval
import dsdtools
import soundfile as sf
import cv2
import pickle


class GenerateMusic:
    def __init__(self, input_wav, encode, decode, enhance):
        super(GenerateMusic, self).__init__()
        self.enhance = enhance
        self.input_wav = input_wav
        self.decode = decode
        self.encode = encode

    def stft_music(self):

        # stft and feature extraction
        fft_size = 512
        hopsamp = fft_size // 8
        stft_full_mixture = librosa.core.stft(self.input_wav, n_fft=fft_size, hop_length=hopsamp, win_length=fft_size)
        stft_full_mixture = stft_full_mixture.transpose()
        stft_len = stft_full_mixture.shape[0]
        stft_mag = abs(stft_full_mixture)
        stft_mag = stft_mag[:, 0:-1]
        stft_mag_out = stft_mag
        if stft_mag.shape[0] % 256 != 0:
            stft_mag = np.concatenate((stft_mag[0:len(stft_mag) // 256 * 256, :],
                                       stft_mag[-256:, :]))
        stft_angle = np.angle(stft_full_mixture)
        stft_mag = np.interp(stft_mag, (stft_mag.min(), stft_mag.max()), (-0, +1))
        stft_mag = stft_mag ** 0.3
        stft_mag = (stft_mag - 0.5) * 2
        stft_images_t = np.reshape(stft_mag, (stft_mag.shape[0] // 256, 256, stft_mag.shape[1]))
        stft_images = np.empty((stft_mag.shape[0] // 256, stft_mag.shape[1], 256))
        counter = 0
        for image in stft_images_t:
            stft_images[counter] = image.transpose()
            counter += 1

        # run through the net and create masks
        stft_images = torch.from_numpy(stft_images).type('torch.FloatTensor')
        masks_mat = np.empty([0])
        with torch.no_grad():
            for i, images in enumerate(stft_images):
                images = Variable(images.cuda())
                images = images.unsqueeze(0)
                images = images.unsqueeze(0)
                outputs = self.decode(self.encode(images))
                if masks_mat.size == 0:
                    masks_mat = outputs
                elif (i == stft_images.shape[0] - 1) and (stft_len % 256 != 0):
                    overlap_size = 256 - stft_len + masks_mat.shape[3]
                    masks_mat[:, :, :, -overlap_size:] = masks_mat[:, :, :, -overlap_size:] / 2 + outputs[:, :, :,
                                                                                                  :overlap_size] / 2
                    masks_mat = torch.cat((masks_mat, outputs[:, :, :, overlap_size:]), 3)
                else:
                    masks_mat = torch.cat((masks_mat, outputs), 3)

        masks_mat = masks_mat.squeeze()
        masks_mat = masks_mat.detach().cpu().numpy()
        masks_mat = masks_mat.transpose()
        return masks_mat, stft_mag_out, stft_angle

    def mask_the_music(self, masks_mat, stft_mag, stft_angle):
        # separate original music array to two complementary music arrays e.g. vocals and accompaniment
        zeros_vec = np.zeros((masks_mat.shape[0], 1))
        masks_mat = masks_mat ** self.enhance
        masks_mat = cv2.bilateralFilter(masks_mat, 3, 10, 10)

        masked_mag_stft = stft_mag * masks_mat
        vocals_mag_stft = stft_mag - masked_mag_stft
        vocals_mag_stft = np.concatenate((vocals_mag_stft, zeros_vec), axis=1)
        interference_mag_stft = np.concatenate((masked_mag_stft, zeros_vec), axis=1)
        vocals_stft = vocals_mag_stft * np.exp(1.0j * stft_angle)
        interference_stft = interference_mag_stft * np.exp(1.0j * stft_angle)

        fft_size = 512
        hopsamp = fft_size // 8
        recon_vocals = librosa.core.istft(vocals_stft.transpose(), hop_length=hopsamp, win_length=fft_size,
                                          dtype='float64')
        recon_inter = librosa.core.istft(interference_stft.transpose(), hop_length=hopsamp, win_length=fft_size,
                                         dtype='float64')
        return recon_vocals, recon_inter

    def forward(self):
        masks_mat, stft_mag_out, stft_angle = self.stft_music()
        recon_vocals, recon_inter = self.mask_the_music(masks_mat, stft_mag_out, stft_angle)
        return recon_vocals, recon_inter


def test_stft_dsd(config, checkpoint_dir, output_folder='./outputs/',
                        method_name='abl_1', target='vocals', is_test=False):
    """
    Testing/evaluating the net. For given generator, pruduces sdr, sir and sar for DSD100 dataset.
    :param config: path to config file
    :param checkpoint_dir: checkpoint_dir: path to generator's saved parameters. In case of evaluating during training,
    path to checkpoints directory.
    :param output_folder: desired output path
    :param method_name: name of method
    :param target: desired target (vocals/drums/bass)
    :param is_test: flag, when running during training should be False. True for total DSD100 evaluation for a given checkpoint.
    :return: stat parameters for the net, added to tensorboard.
    """
    dsd = dsdtools.DB(root_dir='../../data/datasets/music/DSD100')
    tracks = dsd.load_dsd_tracks(subsets='Test')

    config = get_config(config)
    trainer = MUSIC_Trainer(config)
    enhance = 9

    if ~is_test:
        tracks = [tracks[i] for i in [17, 41, 6, 23, 31]]
    last_gen_name = get_model_list(checkpoint_dir, "gen")
    state_dict = torch.load(last_gen_name)
    trainer.gen.load_state_dict(state_dict['gen'])
    trainer.cuda()
    trainer.eval()
    encode, decode = trainer.gen.encode, trainer.gen.decode

    recon_list = []
    sdr_list, sir_list, sar_list = [], [], []
    sdr_list_inter, sir_list_inter, sar_list_inter =[], [], []

    print(method_name)
    for track in tqdm(tracks):
        sample_rate = 20480
        music_array, music_array_ref, music_array_inter = music_track(track, target)
        music_array_samp = librosa.resample(music_array.transpose(), track.rate, sample_rate)

        masker_l = GenerateMusic(music_array_samp[0, :], encode, decode, enhance=enhance)  # default was 8
        recon_vocals_l, recon_inter_l = masker_l.forward()
        masker_r = GenerateMusic(music_array_samp[1, :], encode, decode, enhance=enhance)
        recon_vocals_r, recon_inter_r = masker_r.forward()

        recon_vocals = np.vstack((recon_vocals_l, recon_vocals_r))
        recon_inter = np.vstack((recon_inter_l, recon_inter_r))
        recon_vocals = librosa.resample(recon_vocals, sample_rate, track.rate)
        recon_inter = librosa.resample(recon_inter, sample_rate, track.rate)

        recon_vocals = recon_vocals.transpose()
        recon_inter = recon_inter.transpose()
        recon_list.append(recon_vocals)

        if len(music_array_ref) > len(recon_vocals):
            len_diff = len(music_array_ref) - len(recon_vocals)
            recon_vocals = np.concatenate((recon_vocals, recon_vocals[-len_diff:, :]))
            recon_inter = np.concatenate((recon_inter, recon_inter[-len_diff:, :]))
        elif len(music_array_ref) < len(recon_vocals):
            recon_vocals = recon_vocals[0:len(music_array_ref), :]
            recon_inter = recon_inter[0:len(music_array_ref), :]

        reference_music = np.array([music_array_ref, music_array_inter])
        estimates_music = np.array([recon_vocals, recon_inter])

        sdr_b, _, sir_b, sar_b, _ = museval.metrics.bss_eval_images_framewise(reference_music, estimates_music,
                                                                        window=1323000, hop=661500)
        sdr, sir, sar = sdr_b[0], sir_b[0], sar_b[0]
        sdr_inter, sir_inter, sar_inter = sdr_b[1], sir_b[1], sar_b[1]
        sdr, sir, sar = np.mean(sdr[~np.isnan(sdr)]), np.mean(sir[~np.isnan(sir)]), np.mean(sar[~np.isnan(sar)])
        sdr_inter, sir_inter, sar_inter = np.mean(sdr_inter[~np.isnan(sdr_inter)]), \
                                          np.mean(sir_inter[~np.isnan(sir_inter)]), np.mean(sar_inter[~np.isnan(sar_inter)])

        sdr_list.append(sdr), sir_list.append(sir), sar_list.append(sar)
        sdr_list_inter.append(sdr_inter), sir_list_inter.append(sir_inter), sar_list_inter.append(sar_inter)

    sdr_max = max(sdr_list)
    sdr_max_loc = sdr_list.index(sdr_max)
    sir_max = max(sir_list)
    sir_max_loc = sir_list.index(sir_max)

    sdr_median, sir_median, sar_median = np.median(sdr_list), np.median(sir_list), np.median(sar_list)
    sdr_median_inter, sir_median_inter, sar_median_inter = np.median(sdr_list_inter), \
                                                           np.median(sir_list_inter), np.median(sar_list_inter)

    output_folder = output_folder + method_name
    if not os.path.exists(output_folder):
        print("Creating directory: {}".format(output_folder))
        os.makedirs(output_folder)

    stats = [sdr_median, sir_median, sar_median, sdr_median_inter, sir_median_inter, sar_median_inter]
    stats_name = ['sdr_median', 'sir_median', 'sar_median', 'sdr_median_inter', 'sir_median_inter', 'sar_median_inter']
    with open(os.path.join(output_folder, 'stats_final_test.txt'), 'w') as f:
        for stat_name, stat in zip(stats_name, stats):
            f.write("%s\n" % stat_name), f.write("%s\n" % stat)
            print(stat_name + ': ' + str(stat))
    stats_dic = {'sdr': sdr_list, 'sir': sir_list, 'sar': sar_list, 'sdr_inter': sdr_list_inter,
                 'sir_inter': sir_list_inter, 'sar_inter': sar_list_inter}
    outfile = os.path.join(output_folder, 'final_results')
    save_obj(stats_dic, outfile)

    music_2_write = recon_list[sdr_max_loc]
    music_2_write_sec = recon_list[sir_max_loc]
    sf.write(os.path.join(output_folder, 'best_sdr_iter_' + tracks[sdr_max_loc].filename + '.wav'), music_2_write, track.rate)
    sf.write(os.path.join(output_folder, 'best_sir_iter_' + tracks[sir_max_loc].filename + '.wav'), music_2_write_sec, track.rate)
    return sdr_median, sir_median, sar_median, sdr_max, sir_max


def music_track(track, target):
    music_array = track.audio
    if target == 'vocals':
        music_array_ref = track.targets['vocals'].audio
        music_array_inter = track.targets['accompaniment'].audio
    elif target == 'drums':
        music_array_ref = track.targets['drums'].audio
        music_array_inter = track.targets['vocals'].audio
        music_array_inter += track.targets['bass'].audio
        music_array_inter += track.targets['other'].audio
    elif target == 'bass':
        music_array_ref = track.targets['bass'].audio
        music_array_inter = track.targets['vocals'].audio
        music_array_inter += track.targets['drums'].audio
        music_array_inter += track.targets['other'].audio
    else:
        raise('Not a valid target!')
    return music_array, music_array_ref, music_array_inter


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':

    test_stft_dsd(config='./configs/vocals_new.yaml',
                  checkpoint_dir='./data/singing_outputs/outputs/vocals_new/checkpoints/',
                  output_folder='./outputs/',
                  method_name='method_1', target='vocals', is_test=True)

