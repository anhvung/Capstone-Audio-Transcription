library(tidyverse)
library(ggthemes)
library(RColorBrewer)
library(ggpubr)

data <- read.csv("wer_df.csv")
data <- data[,2:9]
#data$sample_rate <- as.factor(data$sample_rate)
ds.data <- data[,1:4] %>%
  pivot_longer(c("whisper_ds_wer",
                 "wav2vec2_ds_wer",
                 "wav2vec2_4gram_ds_wer"),
               names_to = "model",
               values_to = "WER")
ds.data <- ds.data[-c(13,16),]
ns.data <- data[,5:8] %>%
  pivot_longer(c("whisper_ns_wer",
               "wav2vec2_ns_wer",
               "wav2vec2_4gram_ns_wer"),
             names_to = "model",
             values_to = "WER")

# default natural log
log_breaks <- log(as.numeric(levels(as.factor(data$sample_rate))))
plt_cols <- brewer.pal(3,"Set2")
plt_nms <- c("wav2vec2.0+4gram", "wav2vec2.0",
                     "whisper")

ds.plt <- ds.data %>%
  ggplot(aes(x=log(sample_rate), y=WER, col=model)) + 
  geom_point() +
  geom_point(aes(x=log(500), y=data$whisper_ds_wer[6]), 
             colour="#8DA0CB",
             size=1) +
  geom_line() +
  labs(title="") +
  scale_x_continuous(name="Sample rate (Hz)",
                     breaks=log_breaks, 
                     labels=levels(as.factor(data$sample_rate))) +
  scale_color_manual(name="Model:",
                     labels=plt_nms,
                     values=plt_cols) + 
  theme_stata() +
  theme(legend.title=element_text(vjust=0.5))

ns.plt <- ns.data %>%
  ggplot(aes(x=noise_rate, y=WER, col=model)) + 
  geom_point() +
  geom_line() +
  labs(title="",
       ylab="") +
  scale_x_continuous(name="Noise level",
                     breaks=0:6) +
  scale_color_manual(name="Model:",
                     labels=plt_nms,
                     values=plt_cols) +
  theme_stata() +
  theme(legend.title=element_text(vjust=0.5))

ggarrange(ds.plt, ns.plt, ncol=2, nrow=1,
          common.legend=TRUE, legend="bottom")


ft.data <- data.frame(
  'Language' = c('CN','KR','HE','TE','CN','KR','HE','TE','CN','KR','HE','TE','CN','KR','HE','TE'),
  'WER' = c(19.7,28.4,57.9,40.3,25.78,57.4,60.8,78.27,38.0,32.4,70.6,120.7,100,100,100,100), 
  'Model' =c('whisper-post','whisper-post','whisper-post','whisper-post','wav2vec2-post','wav2vec2-post','wav2vec2-post','wav2vec2-post',
             'whisper-pre','whisper-pre','whisper-pre','whisper-pre','wav2vec2-pre','wav2vec2-pre','wav2vec2-pre','wav2vec2-pre'))
ft.data <- ft.data %>%
  pivot_wider(names_from=Model,
              values_from=WER)
ft.data$`whisper-pre` <- ft.data$`whisper-pre` - ft.data$`whisper-post`
ft.data$`wav2vec2-pre` <- ft.data$`wav2vec2-pre` - ft.data$`wav2vec2-post`
ft.data <- ft.data %>%
  pivot_longer(`whisper-post`:`wav2vec2-pre`, 
               names_to = "Model",
               values_to = "WER")

ft.data$Combo <- ft.data$Model
ft.data <- ft.data %>%
  separate(Model, c("Model", "FT"))
ft.data$Language <- factor(ft.data$Language,
                           levels=c("CN", "KR", "HE", "TE"))
ft.data$Model <- as.factor(ft.data$Model)
levels(ft.data$Model) <- c("w2v2", "whisper") # renamed
ft.data$Combo <- as.factor(ft.data$Combo)
ft.data$Combo <- relevel(ft.data$Combo, "wav2vec2-post")
ft.data$Combo <- relevel(ft.data$Combo, "whisper-pre")
ft.data$Combo <- relevel(ft.data$Combo, "wav2vec2-pre")

plt_nms <- c("pre-w2v2", "pre-whisper",
             "post-w2v2", "post-whisper")
plt_cols <- c("#FC8D62", "#8DA0CB",
              "#9e583d", "#58647f")

ggplot(ft.data, aes(x=Model, y=WER, fill=Combo)) +
  geom_bar(stat="identity") +
  facet_grid(~Language) +
  scale_fill_manual(name="Model:",
                    labels=plt_nms,
                    values=plt_cols) +
  theme_stata() +
  theme(legend.title=element_text(vjust=0.5))
