import colorednoise as cn
from datasets import load_dataset, load_from_disk
from jiwer import wer
import kenlm
import librosa
import numpy as np
import os
import tarfile
import torch
import unicodedata
import urllib.request
from transformers import AutoModelForCTC, AutoProcessor
from whisper.normalizers import EnglishTextNormalizer

# set paths
datasets_path = os.path.join(os.getcwd(), 'datasets') 
predictions_path = os.path.join(os.getcwd(), 'predictions')
predictions_confidence_path = os.path.join(os.getcwd(), 'predictions_confidence')

# create folders if they do not already exist
if not os.path.exists(datasets_path): os.makedirs(datasets_path)
if not os.path.exists(predictions_path): os.makedirs(predictions_path)
if not os.path.exists(predictions_confidence_path): os.makedirs(predictions_confidence_path)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_wav2vec_model(hf_path: str):
    """
    load and return wav2vec tokenizer and model from huggingface
    """
    model = AutoModelForCTC.from_pretrained(hf_path).to(device)
    processor = AutoProcessor.from_pretrained(hf_path)
    return processor, model

def compute_probs(pred_scores, word_dict):
    probs = pred_scores[0, word_dict["start_offset"]: word_dict["end_offset"]]   
    return torch.sum(probs).item() / (len(probs))

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=2, keepdims=True)

def map_to_pred(batch, model, processor):
    """
    predicts transcription
    """
    #tokenize
    input_values = processor(batch["audio"]["array"], return_tensors="pt").input_values
    #take logits
    with torch.no_grad(): logits = model(input_values.to(device)).logits
    #take argmax (find most probable word id)
    predicted_ids = torch.argmax(logits, dim=-1)
    #compute output
    output = processor.batch_decode(logits.numpy(), output_word_offsets=True)
    #compute probs
    probs = softmax(logits.numpy())[0]                      
    probs = {d["word"]:  np.mean(np.max(probs[d['start_offset']:d['end_offset']],axis=1)) for d in output.word_offsets[0]}

    batch['string_pred'] = custom_normalizer(
        output['text'][0], "en")
    batch['tokens_pred'] = [token for token in probs.keys()]
    batch['probs_tokens_pred'] = [probs[token] for token in probs.keys()]
    batch['ground_truth'] = custom_normalizer(
        batch['transcription'], "en")
    batch['wer'] = wer(batch['string_pred'], batch['ground_truth'])

    return batch

def html_display_confidence(prediction_dataset, rows_ids):
    """
    Compute html string with confidence color per token, ground truth and wer
    """

    final_text = ""

    def cstr(s, color='black'):
        return "<text style=color:{}>{}</text>".format(color, s)

    def map_float_rgb(f, m, M):
        rgb = 'rgb({},{},0)'.format(int(255 * (1 - ((f - m) / (M - m)))), int(255 * (f - m) / (M - m)))
        return rgb

    for row_index in rows_ids:
        tokens = prediction_dataset[row_index]['tokens_pred']
        probs_tokens = prediction_dataset[row_index]['probs_tokens_pred']


        min_prob = min(probs_tokens)
        max_prob = max(probs_tokens)


        final_text += "prediction &nbsp  &nbsp :  " + "".join([cstr(s=tokens[idx].replace('Ġ',' '), color=map_float_rgb(probs_tokens[idx], min_prob, max_prob)) for idx in range(len(tokens))]) + "<br>"
        final_text += "ground truth : " + prediction_dataset[row_index]['raw_transcription'] + "<br>"
        final_text += "WER" + 7 * " &nbsp" + ": " + str(round(100 * prediction_dataset[row_index]['wer'], 1)) + "%<br><br>"

    return final_text

def custom_normalizer(text, lang):
    """
    normalizing procedures based on appendix C, Whisper OpenAI paper
    language tokens based on https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
    """
    if lang == 'en':
        normalizer = EnglishTextNormalizer()
        return normalizer(text)
    else:
        # removes [] and () as well as content in-between -- will not work for non-standard brackets, eg: <> or （）, etc
        text = re.sub("[\(\[].*?[\)\]]", "", text)
        text = unicodedata.normalize("NFKC", text)
        ch_text = []
        for ch in text:
            if unicodedata.category(ch)[0] not in ('M', 'P', 'S'):
                ch_text.append(ch)
            else:
                ch_text.append(' ')
        text = ''.join(ch_text)
        text = text.lower()
    # set up for character error rate for languages w/o spaces between words
    if lang in ('zh', 'ja', 'th', 'lo', 'my'):
        text = ' '.join(text)
        # remove spaces between consecutive numbers
        text = re.sub('(?<=\d) (?=\d)', '', text)
    return re.sub(' +', ' ', text)