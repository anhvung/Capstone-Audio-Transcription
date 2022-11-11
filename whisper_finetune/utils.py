import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import sys
sys.path.insert(0, '/home/sivan/asr/whisper')
from utils import custom_normalizer

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

# preprocess
def prepare_dataset(batch, feature_extractor, tokenizer):
    # load resampled audio data
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["transcription"]).input_ids # make sure to encode the transcription column as ground truth
    return batch

# normalizer
def compute_metrics(pred, tokenizer):
    pred_ids = pred.predictions # ids from model.generate
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # normalizer
    # pred_str = custom_normalizer(pred_str, "zh")
    pred_str = [custom_normalizer(str(input_str), "zh") for input_str in pred_str]
    label_str = [custom_normalizer(str(input_str), "zh") for input_str in label_str]

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    print(pred_str[:5], label_str[0:5])
    return {"wer": wer}
