import sys
import unicodedata
import re
from typing import Any, Dict, List, Union

import torch
import evaluate
from dataclasses import dataclass
from transformers import WhisperProcessor
from whisper.normalizers import EnglishTextNormalizer
from transformers import WhisperForConditionalGeneration

metric = evaluate.load("wer")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# Model loader
def load_whisper(path: str, lang: str):
    """
    load and return wav2vec tokenizer and model from huggingface
    """
    processor = WhisperProcessor.from_pretrained(path, language=lang, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(path).to(device)

    return processor, model


# preprocess
def prepare_dataset(batch, feature_extractor, tokenizer):
    # load resampled audio data
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(
        batch["raw_transcription"]).input_ids  # make sure to encode the raw_transcription column as ground truth

    # save normalized transcription for reference
    batch["transcription"] = batch["transcription"]
    return batch


# normalizer

def metrics(lang, tokenizer):
    """
    define metrics according to languages
    """
    def compute_metrics(pred):
        pred_ids = pred.predictions  # ids from model.generate
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # normalizer
        # pred_str = custom_normalizer(pred_str, "zh")
        pred_str_norm = [custom_normalizer(str(input_str), lang) for input_str in pred_str]
        label_str_norm = [custom_normalizer(str(input_str), lang) for input_str in label_str]

        wer_raw = 100 * metric.compute(predictions=pred_str, references=label_str)
        wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

        print(pred_str[:5], label_str[:5])
        return {"wer": wer_raw, "wer_norm": wer}

    return compute_metrics

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
