# Finetuning Whisper

## Approach
We finetuned Whisper base model on four languages using [Fleurs](https://huggingface.co/datasets/google/fleurs) dataset: Chinese, Hebrew, Korean, and Telugu.

We followed the data preprocessing method in Whisper: 
Raw audio input is re-sampled to 16 kHz, broken into 30-second segments (either truncated or zero-padded), then transformed to an 80-channel log magnitude Mel spectrogram with a 25ms window size and a 10ms stride size.
Before feeding features into the encoder, the input is globally scaled to -1 and 1 with approximately zero mean across the pre-training dataset. Different from the unlabelled input in self-supervised Wav2Vec 2.0, Whisper takes pairs of segmented audio features with the subset of the transcript that occurs within each time segment regardless of the inclusion of speech event. 

## Setup and Running
Install required dependencies using:
```angular2html
pip install -r requirements.txt
```

1. `whisper_finetune_demo.ipynb` demonstrates the finetune demo of Chinese ("cmn_hans_cn") in Fleurs.
2. `whisper_finetune_multilingual.ipynb` contains the finetune codes for Hebrew, Korean, and Telugu in Fleurs.
3. `whisper_finetune_eval.ipynb` contains codes for finetuned model evaluation.

All files including checkpoints and models are stored in cloud: `gs://capstone_datasets/fleurs/finetune/"`.



## FAQ
Some useful threads made for solving issues during finetuning:
1. Preprocessing and metrics: https://discuss.huggingface.co/t/seq2seqtrainer-enabled-must-be-a-bool-got-nonetype/25680
2. Encoding tensor size: https://discuss.huggingface.co/t/trainer-runtimeerror-the-size-of-tensor-a-462-must-match-the-size-of-tensor-b-448-at-non-singleton-dimension-1/26010

## References
Part of codes adapted from [HuggingFace community](https://huggingface.co/blog/fine-tune-whisper#fine-tune-whisper-for-multilingual-asr-with-%F0%9F%A4%97-transformers).
```bibtex
@article{radford2022robust,
  title={Robust speech recognition via large-scale weak supervision},
  author={Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2212.04356},
  year={2022}
}
```
