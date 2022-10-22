import librosa
import numpy as np
import colorednoise as cn

#for testing
'''
file_path = r'path\to\file'
signal, sr = librosa.load(file_path, sr = 16000)

add_noise(signal,sample_rate=16000)
'''


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


def down_sample(s, input_sr=16000, output_sr=8000):
    # s: audio input (mono)
    # input_sr: sample rate of s
    # output_sr: output sample rate

    # resample to output_sr
    resampled_s = librosa.resample(s, orig_sr=input_sr, target_sr=output_sr)

    # then re-resample to 16000
    noisy_s = librosa.resample(resampled_s, orig_sr=output_sr, target_sr=16000)
    
    # output should be at 16kHz sample rate
    return noisy_s




## TODO
'''
Check for other non clean datasets (MUSIC?, 'other' librispeech ), check for multi-speaker datasets, multi-lingual datasets.
upload them to buckets
Long form transcription(bonus)
'''















