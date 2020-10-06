# Evaluation of imputed data - how many values are kept for each customer?

library(tidyverse)
data <- read_csv("../data/DSA3101_Hackathon_Data.csv")
names(data) <- c("pid", "date", "cat", "pack_size", "vol", "spend")
data_imputed <-read_csv("../cleaned_data/imputed_data.csv")
data_counts <- data %>% group_by(pid) %>% summarise(count_full = n())
without_imputation_counts <- data %>% 
  filter(pack_size != 0, vol != 0, spend != 0) %>% 
  group_by(pid) %>% summarise(count_without_imputation = n())
imputed_counts <- data_imputed %>% group_by(pid) %>% summarise(count_with_imputation = n())

kept_data <- data_counts %>% 
  left_join(without_imputation_counts, by="pid") %>%
  left_join(imputed_counts, by="pid") %>%
  mutate(percent_kept_before = count_without_imputation * 100 / count_full,
         percent_kept_after  = count_with_imputation * 100 / count_full) %>%
  select(pid, percent_kept_before, percent_kept_after)
  
improvements <- kept_data %>% filter(percent_kept_before < 90) %>% mutate(improvement = percent_kept_after - percent_kept_before)

kept_data %>% pivot_longer(c(percent_kept_before, percent_kept_after), 
                           names_to = "key", values_to = "percent") %>% 
  ggplot() + 
  geom_boxplot(aes(x = percent, color = key)) + 
  theme(axis.text.y = element_blank(), axis.line.y = element_blank())

improvements$improvement %>% summary()

nrow(improvements %>% filter(percent_kept_after < 90))