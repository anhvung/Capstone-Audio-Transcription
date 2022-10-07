import librosa


#for testing
'''
file_path = r'path\to\file'
signal, sr = librosa.load(file_path, sr = 16000)

add_noise(signal,sample_rate=16000)
'''


def add_noise(s,sample_rate=16000,noise_db=-40,noise_type='white'):
    # s: audio input (mono)
    # noise_db rms value of the added noiseo signal
    # sample rate: sample rate of s
    # type: white, pink, gaussing
    
    
    # output should be at 16kHz sample rate
    return noisy_s


def add_signals(s,sample_rate=16000,back_s,back_sample_rate=160000,noise_db=-12):
    # s: audio input (mono)
    # noise_db rms value of the added noiseo signal
    # back_s: brckgrnd audio 
    # sample rate: sample rate of s
    # noise_db: lower the backgrnd signal by noise_db db
    
    
    
    # output should be at 16kHz sample rate
    return noisy_s


def add_signals(s,sample_rate=16000,output_sr=8000):
    # s: audio input (mono)
    # sample rate: sample rate of s
    # output_sr: output sample rate
    
    
    # resample to output_sr

    # then re-resample to 160000
    
    # output should be at 16kHz sample rate
    return noisy_s



## TODO
'''
Check for other non clean datasets (MUSIC?, 'other' librispeech ), check for multi-speaker datasets, multi-lingual datasets.
upload them to buckets
Long form transcription(bonus)
'''















