# Title     : TODO
# Objective : TODO
# Created by: mati
# Created on: 08.11.2020
# Glove pretrained model: http://nlp.stanford.edu/data/glove.6B.zip

library(SnowballC)
library(tidytext)
library(dplyr)
library(text2vec)
library(data.table)

load_train <- function () {
  read.csv('classification/data/winemag-data-130k-v2.csv', header = TRUE)
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

discretize_review <- function (df) {
  df$sentiment <- df$points > 90
  return(df)
}

load_glove <- function () {
  if (!file.exists('classification/data/glove.6B.50d.txt')) {
    download.file('http://nlp.stanford.edu/data/glove.6B.zip',destfile = 'classification/data/glove.6B.zip')
    unzip('classification/data/glove.6B.zip')
  }
  glove <- fread('classification/data/glove.6B.50d.txt', data.table = F,  encoding = 'UTF-8')
  names(glove) <- c('word', paste('dim', 1:50, sep = '_'))
  return(glove)
}

embed_doc <- function (entry, glove) {
  # split strings into vector
  v <- unlist(strsplit(entry[3], ' '))
  # only words with 2+ characters
  v <- unique(v[grepl('..+', v)])
  # find glove repr
  embed <- lapply(v, FUN = function (x) as.vector(glove[glove$word == x, -1]))
  # create matrix of word vectors (1 row is 1 word)
  m <- matrix(unlist(embed), nrow=length(embed), byrow=TRUE)
  # calculate mean of each dim
  colSums(m) / length(v)
}

embed_description <- function (df, glove) {
  df$embeded <- apply(df,1, function (x) embed_doc(x, glove))
  return(df)
}


# example
#df <- load_train()
#glove <- load_glove()
#df <- clean_text(df, 'description')
#df <- discretize_review(df)
#df <- embed_description(df[1:20,], glove)
#head(df)

