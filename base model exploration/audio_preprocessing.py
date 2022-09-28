import librosa
import soundfile as sf
import librosa.display
import numpy as np
# To display audio in notebook output
from IPython.display import Audio as Audio_display
from IPython.core.display import display


def normalize_audio(audio):
    # for loudness normalization, migth use lufs or rms metrics if necessary
    audio=librosa.util.normalize(audio)
    return audio

def noise_removal(audio, sample_rate = 16000):
    # bckgrnd noise removal https://librosa.org/librosa_gallery/auto_examples/plot_vocal_separation.html for more details
    print('removing noise')
    S_filter = librosa.decompose.nn_filter(audio,
                                           aggregate=np.median,
                                           metric='cosine',
                                           width=int(librosa.time_to_frames(2, sr=sample_rate)))

    S_filter = np.minimum(audio, S_filter)

    margin_i, margin_v = 2, 10
    power = 2

    mask_i = librosa.util.softmask(S_filter,
                                   margin_i * (audio - S_filter),
                                   power=power)

    mask_v = librosa.util.softmask(audio - S_filter,
                                   margin_v * S_filter,
                                   power=power)

    S_foreground = mask_v * audio
    #S_background = mask_i * audio
    return S_foreground
    
def listen(audio, sampling_rate = 16000):
    display(Audio_display(audio, rate=sampling_rate, autoplay=False))
   
    
def stats(audio, sampling_rate = 16000):
    print('avg RMS', np.mean(librosa.feature.rms(audio,sampling_rate)))
    S = librosa.feature.melspectrogram(y=audio, sr=sampling_rate, n_mels=128,
                                    fmax=8000)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                             y_axis='mel', sr=sampling_rate,
                             fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    
    #print('mel spectrogram', librosa.feature.melspectrogram(audio, sampling_rate))
def read_audio(file_path,normalize=True, remove_noise=False,preview=True,debug=True):
    #read the file
    audio, samplerate = sf.read(file_path)
    #make it 1-D
    if len(audio.shape) > 1: 
        audio = audio[:,0] + audio[:,1]
    #Resample to 16khz
    if samplerate != 16000:
        audio = librosa.resample(audio, samplerate, 16000)
    
    if debug:
        stats(audio)
    
    if normalize:
        audio=normalize_audio(audio)
    
    if remove_noise:
        audio=noise_removal(audio)
     
    if preview:
        listen(audio)
        
    return audio
       
    