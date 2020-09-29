library(tidyverse)
data <- read_csv("../data/DSA3101_Hackathon_Data.csv")
correct <- filter(data, Volume != 0)
missing <- filter(data, Volume == 0)
imputed <- missing %>% 
  left_join(correct, by=c("Category", "Pack Size", "Spend")) %>%
  filter(!is.na(Volume.y))

imputed <- imputed %>% select(pid=`Panel ID.x`, 
                              date=Date.x, 
                              cat=Category, 
                              pack_size=`Pack Size`, 
                              spend=Spend, 
                              matched_date=Date.y, 
                              matched_vol=Volume.y)
consistent_imputers <- imputed %>% 
  group_by(pid, date, cat, pack_size, spend) %>% 
  summarise(consistent=all(matched_vol == first(matched_vol)),
            first=first(matched_vol))

true_match <- consistent_imputers %>% filter(consistent) %>%
  select(pid, date, cat, pack_size, vol = first, spend)

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

inconsistent_imputers <- imputed %>% anti_join(true_match) %>% 
  group_by(pid, date, cat, pack_size, spend) %>% 
  summarise(sd=sd(matched_vol),
            mode=Mode(matched_vol))

fuzzy_matched <-inconsistent_imputers %>% filter(sd < 0.5) %>%
  select(pid, date, cat, pack_size, vol = mode, spend)

names(correct) <- c("pid", "date", "cat", "pack_size", "vol", "spend")

data_imputed <- rbind(true_match, fuzzy_matched, correct)
write_csv(data_imputed, "../cleaned_data/imputed_data.csv")
