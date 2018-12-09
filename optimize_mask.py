from utils import get_config
from trainer import MUSIC_Trainer
from tqdm import tqdm
import numpy as np
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import torch
import librosa
import museval
import dsdtools
import scipy
from test import GenerateMusic, music_track


def optimize_mask(enhance0):

    config = '/home/moshe/ext/code/MUSIC/configs/music2vocals_stft_12.yaml'
    checkpoint_dir = '/home/moshe/ext/data/munit_outputs/music2vocal_stft/outputs/music2vocals_stft_12/checkpoints/gen_00285000.pt'
    target = 'vocals'
    dsd = dsdtools.DB(root_dir='../../data/datasets/music/DSD100')
    tracks = dsd.load_dsd_tracks(subsets='Test')

    config = get_config(config)
    trainer = MUSIC_Trainer(config)

    last_gen_name = checkpoint_dir
    state_dict = torch.load(last_gen_name)
    trainer.gen.load_state_dict(state_dict['gen'])
    trainer.cuda()
    trainer.eval()
    encode = trainer.gen.encode  # encode function
    decode = trainer.gen.decode  # decode function

    recon_list = []
    sdr_list, sir_list, sar_list = [], [], []

    for track in tqdm(tracks):
        sample_rate = 20480
        music_array, music_array_ref, music_array_inter = music_track(track, target)
        music_array_samp = librosa.resample(music_array.transpose(), track.rate, sample_rate)

        masker_l = GenerateMusic(music_array_samp[0, :], encode, decode, enhance=enhance0[0])
        recon_vocals_l, recon_inter_l = masker_l.forward()
        masker_r = GenerateMusic(music_array_samp[1, :], encode, decode, enhance=enhance0[0])
        recon_vocals_r, recon_inter_r = masker_r.forward()

        recon_vocals = np.vstack((recon_vocals_l, recon_vocals_r))
        recon_inter = np.vstack((recon_inter_l, recon_inter_r))
        recon_vocals = librosa.resample(recon_vocals, sample_rate, track.rate)
        recon_inter = librosa.resample(recon_inter, sample_rate, track.rate)

        recon_list.append(recon_vocals)
        recon_vocals = recon_vocals.transpose()
        recon_inter = recon_inter.transpose()

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
        sdr, sir, sar = np.mean(sdr[~np.isnan(sdr)]), np.mean(sir[~np.isnan(sir)]), np.mean(sar[~np.isnan(sar)])

        sdr_list.append(sdr), sir_list.append(sir), sar_list.append(sar)

    sdr_median, sir_median, sar_median = np.median(sdr_list), np.median(sir_list), np.median(sar_list)
    print('mask size ', enhance0[0])
    print('sdr_median ', sdr_median)
    print('sir_median ', sir_median)
    print('sar_list ', sar_list)

    return -sdr_median - sir_median * 0.1


def run_optimization():

    print('begin with full optimization, 200 iterations!')
    enhance0 = 12
    res = scipy.optimize.minimize(optimize_mask, enhance0, method='Nelder-Mead', options={'maxiter': 200, 'disp': True})
    # res = scipy.optimize.minimize(optimize_mask, enhance0, method='Nelder-Mead', options={'disp': True}, tol=1e-6)
    print(res.x)
    print(res.fun)


if __name__ == '__main__':

    run_optimization()

