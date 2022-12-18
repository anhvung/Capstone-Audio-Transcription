# Audio Transcription with WAV2VEC 2.0 

### Group members Name UNI 
- Anh-Vu Nguyen an3078 (Team captain)
- Alexandria Guo ag4475
- Julia Wang jw4169
- Maxwell Zhou mtz2110
- Sivan Ding sd3507
- Antonin Vidon av3023

Emails  &lt;UNI&gt; @ columbia.edu

**Accenture mentor & co-mentors:** Dr Bhushan, Surajit Sen, Priyanka Pandey

**Instructor/CA:** Prof. Sining Chen, Aayush Verma

Exploration of an open source-based ASR (Automatic Speech Recognition) solution that enables custom training for Acoustic as well Language model. The solution will leverage publicly available pre-trained acoustic model that can be fine-tuned with small amount of domain specific data. 

We will start to explore Meta AI’s wav2vec 2.0 Framework for acoustic model.
Further to improve the accuracy, we will experiment (training/finetuning) with a language model based on n-gram with KenLM or Transformer architecture. Once the models are trained, evaluate the models using KPI Word-Error-Rate (WER) by decoding audio data with acoustic in combination with language model.

**Directory tree**
```
│   .gitignore
│   CONTRIBUTING.md
│   README.md
│   requirements.txt
│
├───.ipynb_checkpoints
├───confidence_scores
│       analyze.ipynb
│       generate.ipynb
│       utils.py
│
├───data preprocessing
│   └───.ipynb_checkpoints
├───documentation
│       GCP.txt
│       README.md
│
├───EDA
│       capstone_EDA.ipynb
│
├───preprocessing
│       audio_preprocess.py
│       preprocess examples.ipynb
│       README.md
│
├───wav2vec
│   │   add_noise.ipynb
│   │   downsample.ipynb
│   │   KR_wav2vec2_XLS_R_finetune.ipynb
│   │   plots.ipynb
│   │   template.ipynb
│   │   utils.py
│   │   wav2vec4g_accent.ipynb
│   │   wav2vec4g_experiments.ipynb
│   │   wav2vec4g_noisy.ipynb
│   │   wav2vec_accent.ipynb
│   │   wav2vec_languages.ipynb
│   │   wav2vec_noisy.ipynb
│   │
│   └───.ipynb_checkpoints
│           wav2vec4g_experiments-checkpoint.ipynb
│           wav2vec_confidence_score-checkpoint.ipynb
│
├───wav2vec_finetune
│       wav2vec_finetune_hebrew.ipynb
│       wav2vec_finetune_hebrew_eval.ipynb
│       wav2vec_hebrew_eval.ipynb
│
├───whisper
│   │   audio.mp3
│   │   requirements.txt
│   │   robust_comp_plt.ipynb
│   │   robust_downsample.ipynb
│   │   robust_noise.ipynb
│   │   utils.py
│   │   wer_df.csv
│   │   whisper_intro.ipynb
│   │   whisper_intro_noise.ipynb
│   │
│   └───.ipynb_checkpoints
│           robust_noise-checkpoint.ipynb
│           whisper_intro-checkpoint.ipynb
│
└───whisper_finetune
        README.md
        requirements.txt
        utils.py
        whisper_finetune_demo.ipynb
        whisper_finetune_eval.ipynb
        whisper_finetune_multilingual.ipynb
```



