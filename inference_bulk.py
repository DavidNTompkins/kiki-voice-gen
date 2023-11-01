import librosa
import matplotlib.pyplot as plt
import scipy
from scipy.io.wavfile import write as wav_write

import os
import json
import math
import numpy as np

import requests
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import langdetect

from scipy.io.wavfile import write
import re
from scipy import signal

'''
from phonemizer.backend.espeak.wrapper import EspeakWrapper
_ESPEAK_LIBRARY = 'C:\Program Files\eSpeak NG\libespeak-ng.dll'
EspeakWrapper.set_library(_ESPEAK_LIBRARY)
'''
# check device
if torch.cuda.is_available() is True:
    device = "cuda:0"
else:
    device = "cpu"


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def langdetector(text):  # from PolyLangVITS
    try:
        lang = langdetect.detect(text).lower()
        if lang == 'ko':
            return f'[KO]{text}[KO]'
        elif lang == 'ja':
            return f'[JA]{text}[JA]'
        elif lang == 'en':
            return f'[EN]{text}[EN]'
        elif lang == 'zh-cn':
            return f'[ZH]{text}[ZH]'
        else:
            return text
    except Exception as e:
        return text
    

def synthesize_voice_for_checkpoint(checkpoint_path, text_input, checkpoint):  # added checkpoint parameter
    """Synthesize voice for a given checkpoint and text input."""
    _ = utils.load_checkpoint(checkpoint_path, net_g, None)
    vcss(text_input, checkpoint)  # passing the checkpoint here

# Define the desired checkpoints
checkpoints = [ 
    2_135_000
]


def vcss(inputstr, checkpoint): # added checkpoint parameter
    fltstr = re.sub(r"[\[\]\(\)\{\}]", "", inputstr)
    fltstr = langdetector(fltstr) 
    stn_tst = get_text(fltstr, hps)

    speed = 1
    output_dir = 'output'
    sid = 0
    with torch.no_grad():
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1 / speed)[0][0, 0].data.cpu().float().numpy()
    # Modify the filename to include the checkpoint number
    write(f'./{output_dir}/output_{sid}_chkpt_{checkpoint}_{input_text}.wav', hps.data.sampling_rate, audio)
    print(f'./{output_dir}/output_{sid}_chkpt_{checkpoint}.wav Generated!')


def vcms(inputstr, sid):
    fltstr = re.sub(r"[\[\]\(\)\{\}]", "", inputstr)
    fltstr = langdetector(fltstr)
    stn_tst = get_text(fltstr, hps)

    speed = 1
    output_dir = 'output'
    with torch.no_grad():
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        sid = torch.LongTensor([sid]).to(device)
        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1 / speed)[0][
            0, 0].data.cpu().float().numpy()
    write(f'./{output_dir}/output_{sid}_{input_text}.wav', hps.data.sampling_rate, audio)
    print(f'./{output_dir}/output_{sid}.wav Generated!')




hps = utils.get_hparams_from_file("./configs/mini_mb_istft_vits2_base.json")

if "use_mel_posterior_encoder" in hps.model.keys() and hps.model.use_mel_posterior_encoder == True:
    print("Using mel posterior encoder for VITS2")
    posterior_channels = 80  # vits2
    hps.data.use_mel_posterior_encoder = True
else:
    print("Using lin posterior encoder for VITS1")
    posterior_channels = hps.data.filter_length // 2 + 1
    hps.data.use_mel_posterior_encoder = False

net_g = SynthesizerTrn(
    len(symbols),
    posterior_channels,
    hps.train.segment_size // hps.data.hop_length,
    # n_speakers=hps.data.n_speakers, #- for multi speaker
    **hps.model).to(device)
_ = net_g.eval()

_ = utils.load_checkpoint("./logs/ayaka_mini/G_2135000.pth", net_g, None)

input_texts = ["This is my final voice.",
              "I think that I sound much better now.",
              "Thanks for listening to my recording"

]
for chkpt in checkpoints:
    checkpoint_path = f"./logs/ayaka_mini/G_{chkpt}.pth"
    for input_text in input_texts:
        synthesize_voice_for_checkpoint(checkpoint_path, input_text, chkpt)  # passing the checkpoint here

audios = []

for chkpt in checkpoints:
    filename = f'./output/output_0_chkpt_{chkpt}.wav'
    sr, y = scipy.io.wavfile.read(filename)
    audios.append(y)

combined_audio = np.concatenate(audios)

# Save the concatenated audio
combined_filename = './output/combined_output.wav'
wav_write(combined_filename, sr, combined_audio)
print(f'{combined_filename} Generated!')