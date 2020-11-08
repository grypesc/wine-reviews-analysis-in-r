# Title     : TODO
# Objective : TODO
# Created by: mati
# Created on: 08.11.2020

#install.packages('SnowballC')
#install.packages('tidytext')
#install.packages('dplyr')
#library(SnowballC)
#library(tidytext)
#library(dplyr)

load_train <- function () {
  read.csv('../../data/winemag-data-130k-v2.csv', header = TRUE)
}

remove_outliers <- function(df, col) {
  df[!df[[col]] %in% boxplot.stats(df[[col]])$out, ]
}

clean_text <- function (df, col) {
  text <- df[[col]]
  text <- tolower(text)
  # mentions
  text <- gsub('@\\w+', '', text)
  # links
  text <- gsub('https?://.+', '', text)
  # digits
  text <- gsub('\\d+\\w*\\d*', '', text)
  text <- gsub('#\\w+', '', text)
  text <- gsub('[^\x01-\x7F]', '', text)
  # punctuation
  text <- gsub('[[:punct:]]', ' ', text)
  # new lines
  text <- gsub('\n', ' ', text)
  text <- gsub('^\\s+', '', text)
  text <- gsub('\\s+$', '', text)
  text <- gsub('[ |\t]+', ' ', text)
  df[[col]] <- text
  df
}

# TODO
remove_stopwords <- function (df, col) {
  data('stop_words')
  mutate_at(df, de = as.character(text)) %>%
    select(text) %>%
    unnest_tokens("word", text)
  text <- df[[col]]
  stopwords_regex <- paste(stop_words['word'], collapse = '\\b|\\b')
  stopwords_regex <- paste0('\\b', stopwords_regex, '\\b')
  text <- text %>% filter(!word %in% stop_words$word)
  df[[col]] <- text
  df
}

cleaned_train <- function () {
  df <- load_train()
  df <- remove_outliers(df, 'price')
  clean_text(df, 'description')
}

df <- load_train()
df <- clean_text(df, 'description')
df <- remove_stopwords(df, 'description')
head(df)

