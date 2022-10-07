library(tidyverse)
library(lubridate)

data <- read_csv("speakers.csv", col_names=F)
colnames(data) <- c("ID", "SEX", "SUBSET", "TIME", "NAME")
data$ID <- as.integer(data$ID)
data$SEX <- as.factor(data$SEX)
data$SUBSET <- as.factor(data$SUBSET)
data$TIME <- period_to_seconds(lubridate::hms(data$TIME))
data$NAME <- as.character(data$NAME)

for (subdata in levels(data$SUBSET)) {
  print(subdata)
  data %>% 
    subset(SUBSET == subdata) %>%
    group_by(SEX) %>%
    summarise(TotalMin = sum(TIME)) %>%
    ungroup() %>% 
    print()
}
