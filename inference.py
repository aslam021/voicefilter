import os
import glob
import torch
import librosa
import argparse
import numpy as np
import soundfile as sf

from utils.audio import Audio
from utils.hparams import HParam
from model.model import VoiceFilter
from model.embedder import SpeechEmbedder


def main(args, hp):
    with torch.no_grad():
        model = VoiceFilter(hp).cuda()
        chkpt_model = torch.load(args.checkpoint_path)['model']
        model.load_state_dict(chkpt_model)
        model.eval()

        embedder = SpeechEmbedder(hp).cuda()
        chkpt_embed = torch.load(args.embedder_path)
        embedder.load_state_dict(chkpt_embed)
        embedder.eval()

        audio = Audio(hp)
        dvec_wav, _ = librosa.load(args.reference_file, sr=16000)
        dvec_mel = audio.get_mel(dvec_wav)
        dvec_mel = torch.from_numpy(dvec_mel).float().cuda()
        dvec = embedder(dvec_mel)
        dvec = dvec.unsqueeze(0)

        mixed_wav, _ = librosa.load(args.mixed_file, sr=16000)
        
        mag, phase = audio.wav2spec(mixed_wav)
        mag = torch.from_numpy(mag).float().cuda()

        mag = mag.unsqueeze(0)
        mask = model(mag, dvec)
        est_mag = mag * mask

        est_mag = est_mag[0].cpu().detach().numpy()
        est_wav = audio.spec2wav(est_mag, phase)
        
        # if mixed is biggest than estimation wav we need pad with zeros because is expected that this part is silence
        len_est = len(est_wav)
        len_mixed = len(mixed_wav)
        if len_est < len_mixed:
            est_wav = np.pad(est_wav, (0, len(mixed_wav)-len(est_wav)), 'constant', constant_values=(0, 0))
        
        #to get inverse
        inv_wav = mixed_wav - est_wav
        
        os.makedirs(args.out_dir, exist_ok=True)
        
        out_path = os.path.join(args.out_dir, 'mixed.wav')
        # librosa.output.write_wav(out_path, mixed_wav, sr=16000)
        sf.write(out_path, inv_wav, 16000, 'PCM_24')
        
        out_path = os.path.join(args.out_dir, 'reference.wav')
        # librosa.output.write_wav(out_path, dvec_wav, sr=16000)
        sf.write(out_path, dvec_wav, 16000, 'PCM_24')
        
        out_path = os.path.join(args.out_dir, 'test_takers.wav')
        # librosa.output.write_wav(out_path, est_wav, sr=16000)
        sf.write(out_path, est_wav, 16000, 'PCM_24')
        
        out_path = os.path.join(args.out_dir, 'environment.wav')
        # librosa.output.write_wav(out_path, inv_wav, sr=16000)
        sf.write(out_path, inv_wav, 16000, 'PCM_24')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-e', '--embedder_path', type=str, required=True,
                        help="path of embedder model pt file")
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file")
    parser.add_argument('-m', '--mixed_file', type=str, required=True,
                        help='path of mixed wav file')
    parser.add_argument('-r', '--reference_file', type=str, required=True,
                        help='path of reference wav file')
    parser.add_argument('-o', '--out_dir', type=str, required=True,
                        help='directory of output')

    args = parser.parse_args()

    hp = HParam(args.config)

    main(args, hp)
