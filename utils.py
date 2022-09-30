from datasets import load_dataset
from jiwer import wer
import librosa
import nltk
import os
import tarfile
import torch
import urllib.request
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
nltk.download('punkt')

# set paths
datasets_path = os.path.join(os.getcwd(), 'datasets') 
# create folders if they do not already exist
if not os.path.exists(datasets_path): os.makedirs(datasets_path)
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
    batch['txt'] = txt
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
    batch["transcription"] = transcription
    return batch