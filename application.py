from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
import os

import librosa
import matplotlib.pyplot as plt

import os
import json
import math

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
import uuid

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


def vcss(inputstr): 
    fltstr = re.sub(r"[\[\]\(\)\{\}]", "", inputstr)
    fltstr = langdetector(fltstr)
    stn_tst = get_text(fltstr, hps)

    speed = 1

    with torch.no_grad():
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1 / speed)[0][
                0, 0].data.cpu().float().numpy()
    
    return audio



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
    write(f'./{output_dir}/output_{sid}.wav', hps.data.sampling_rate, audio)
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



application = Flask(__name__)

# Allow only your website domain
CORS(app, resources={r"/synthesize": {"origins": "https://narratorbackendtesting.davidtompkins3.repl.co"}})

@app.route('/synthesize', methods=['POST'])
@cross_origin(origin='https://narratorbackendtesting.davidtompkins3.repl.co')
def synthesize():
    text_input = request.json.get('text', None)
    if not text_input:
        return jsonify({'error': 'No text provided'}), 400

    audio_data = vcss(text_input)
    
    # Convert the audio data to WAV format in memory
    buffer = BytesIO()
    wav_write(buffer, hps.data.sampling_rate, audio_data)
    
    # Send the audio data as a response
    buffer.seek(0)
    return send_file(buffer, mimetype="audio/wav", as_attachment=True, attachment_filename="output.wav")

# Modify vcss to return the output file path
def vcss(inputstr): 
        fltstr = re.sub(r"[\[\]\(\)\{\}]", "", inputstr)
    fltstr = langdetector(fltstr) # can be optional depending on cleaner you use
    stn_tst = get_text(fltstr, hps)

    speed = 1
    output_dir = 'output'
    sid = 0
    with torch.no_grad():
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1 / speed)[0][
                0, 0].data.cpu().float().numpy()
    write(f'./{output_dir}/output_{sid}.wav', hps.data.sampling_rate, audio)
    print(f'./{output_dir}/output_{sid}.wav Generated!')
    
    output_path = f'./{output_dir}/output_{sid}.wav'
    write(output_path, hps.data.sampling_rate, audio)
    return os.path.basename(output_path)

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000)
