# Confidence Scores

This folder contains all the analysis related to token confidence scores. We use the english subset of fleurs and the prediction of the following two models : wav2vec 2.0 + 4-grams and whisper.

**Directory tree**
```
├── confidence_scores
│   ├── README.md
│   ├── analyze.ipynb # all the analysis
│   ├── confidence_scores_sentences.png
│   ├── generate.ipynb # prediction notebook
│   ├── predictions # predictions of models, ignored in github
│   │   ├── fleurs_en_wav2vec
│   │   │   ├── dataset.arrow
│   │   │   ├── dataset_info.json
│   │   │   └── state.json
│   │   └── fleurs_en_whisper
│   │       ├── dataset.arrow
│   │       ├── dataset_info.json
│   │       └── state.json
│   └── utils.py # all utils
```
