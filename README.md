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

The capstone project aims to test and improve the performance of open-source ASR (Automatic Speech Recognition) models: Facebook AI’s wav2vec 2.0, wav2vec 2.0 with a 4-gram language model, and OpenAI’s Whisper. We obtain model performance baselines, measured by word error rate (WER), by transcribing audio data from the Librispeech corpus, and then test the models’ robustness on noisy and downsampled versions of the data. We also perform tests to obtain transcription accuracy on English audio recorded by speakers with different accents and also on recordings in different languages (using wav2vec 2.0 XLSR). We build upon these models by finetuning them on audio data from the Fleurs dataset. The resulting models from our custom finetuning pipelines significantly improve WER performance over their pre-trained counterparts: finetuned wav2vec 2.0 XLSR shows an average of 44.43% (at best, 74.2%) relative reduction in WER and fine-tuned Whisper achieves an average of 36.28% (at best, 66.6%) reduction in WER compared to their base models.


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
│       audio_EDA.ipynb
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



