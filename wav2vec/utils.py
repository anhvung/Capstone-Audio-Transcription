import colorednoise as cn
from datasets import load_dataset, load_from_disk
from jiwer import wer
import librosa
import numpy as np
import os
import tarfile
import torch
import urllib.request
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# set paths
datasets_path = os.path.join(os.getcwd(), 'datasets') 
predictions_path = os.path.join(os.getcwd(), 'predictions')
# create folders if they do not already exist
if not os.path.exists(datasets_path): os.makedirs(datasets_path)
if not os.path.exists(predictions_path): os.makedirs(predictions_path)
# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# now useless as we download directly from bucket, keeping it here for now
def download_and_extract_dataset_from_url(url: str, datasets_path: str = datasets_path):
    """
    downloads and extracts dataset from url into datasets_path/
    """
    temp = os.path.join(datasets_path, url.split('/')[-1])
    print('downloading dataset...')
    urllib.request.urlretrieve(url, temp)
    print('extracting data...')
    file = tarfile.open(temp)
    file.extractall(datasets_path)
    file.close()
    os.remove(temp)
    print('done.')
    
def map_to_ground_truth(batch):
    """
    inserts ground truth in dataset
    """
    transcription_file_path = batch['audio']['path'][:-10] + '.trans.txt'
    f = open(transcription_file_path, 'r')
    lines= str.splitlines(f.read())
    txt=lines[int(batch['audio']['path'][-7:-5])].split(' ', 1)[1]
    batch['ground_truth'] = txt
    return batch

def load_wav2vec_model(hf_path: str):
    """
    load and return wav2vec tokenizer and model from huggingface
    """
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(hf_path)
    model = Wav2Vec2ForCTC.from_pretrained(hf_path).to(device)
    return tokenizer, model

def map_to_pred(batch, model, tokenizer):
    """
    predicts transcription
    """
    #tokenize
    input_values = tokenizer(batch["audio"]["array"], return_tensors="pt").input_values
    #take logits
    logits = model(input_values.to(device)).logits
    #take argmax (find most probable word id)
    predicted_ids = torch.argmax(logits, dim=-1)
    #get the words from the predicted word ids
    transcription = tokenizer.decode(predicted_ids[0])
    #save logits and transcription
    batch["logits"] = logits.cpu().detach().numpy()
    batch["transcription"] = transcription
    return batch

def down_sample(s, sample_rate=16000, output_sr=8000):
    # s: audio input (mono)
    # sample rate: sample rate of s
    # output_sr: output sample rate

    # resample to output_sr
    resampled_s = librosa.resample(s, orig_sr=sample_rate, target_sr=output_sr)

    # then re-resample to 16000
    noisy_s = librosa.resample(resampled_s, orig_sr=output_sr, target_sr=16000)
    
    # output should be at 16kHz sample rate
    return noisy_s


def add_noise(s, sample_rate=16000, noise_percentage_factor = .01, noise_type='white'):
    # s: audio input (mono)
    # sample rate: sample rate of s
    # noise_percentage_factor, percentage scale of added noise added
    # type: white, pink, brown

    if noise_type == 'white':
        beta = 0

    elif noise_type == 'pink':
        beta = 1

    elif noise_type == 'brown':
        beta = 2

    noise = cn.powerlaw_psd_gaussian(beta, s.size)
    noisy_s = s + noise * noise_percentage_factor

    # output should be at 16kHz sample rate
    if sample_rate != 16000:
        noisy_s = librosa.resample(noisy_s, orig_sr = sample_rate, target_sr=16000)

    return noisy_s


def add_signals(s, back_s, sample_rate=16000, back_sample_rate=16000, noise_db=-12):
    # s: audio input (mono)
    # back_s: brckgrnd audio
    # sample rate: sample rate of s
    # noise_db: lower the backgrnd signal by noise_db db


    # make sure both signals have same 16kHz sample rate
    if sample_rate != 16000:
        s = librosa.resample(s, orig_sr=sample_rate, target_sr=16000)

    if back_sample_rate != 16000:
        back_s = librosa.resample(back_s, orig_sr=back_sample_rate, target_sr=16000)

    if s.size > back_s.size:
        back_s = librosa.util.pad_center(back_s, size=s.size)

    elif s.size < back_s.size:
        s = librosa.util.pad_center(s, size=back_s.size)

    # lower background signal by noise_db
    noise_amp = librosa.db_to_amplitude(noise_db)
    lower_back_s = back_s - noise_amp

    # add background noise to sound clip
    noisy_s = s + back_s

    # output should be at 16kHz sample rate
    return noisy_s

def map_to_downsampled(batch, input_sr=16000, output_sr=8000):
    """
    map to downsampled audio array
    """
    batch['audio']['array'] = down_sample(batch['audio']['array'], input_sr, output_sr)
    batch['audio']['sampling_rate'] = output_sr
    return batch

def map_to_noisy(batch, sample_rate=16000, noise_percentage_factor = .01, noise_type='white'):
    """
    map to downsampled audio array
    """
    batch['audio']['array'] = add_noise(batch['audio']['array'], sample_rate=sample_rate, noise_percentage_factor = noise_percentage_factor, noise_type=noise_type)
    return batch