# Title     : TODO
# Objective : TODO
# Created by: mati
# Created on: 08.11.2020
# Glove pretrained model: http://nlp.stanford.edu/data/glove.6B.zip

library(dplyr)
library(data.table)
library(readr)

load_wines <- function () {
  wine <- read_csv('data/winemag-data_first150k.csv')
  wine <- subset(wine, select = c(X1, points, price, description))
  wine$sentiment <- ifelse(wine$points>90, 1, 0)
  setDT(wine)
  setkey(wine, X1)
  return(wine[1:200,])
}

load_glove <- function (dims) {
  # dims - glove dataset dimensions 50/100/200/300
  file <- sprintf('data/glove.6B.%dd.txt', dims)
  if (!file.exists(file)) {
    download.file('http://nlp.stanford.edu/data/glove.6B.zip',destfile = 'data/glove.6B.zip')
    unzip('classification/data/glove.6B.zip')
  }
  glove <- fread(file, data.table=F, encoding='UTF-8')
  names(glove) <- c('word', paste('dim', 1:dims, sep = '_'))
  return(glove)
}

clean_description <- function (df) {
  text <- df$description
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
  df$description <- text
  df
}

embed_doc <- function (doc, glove) {
  # split strings into vector
  v <- unlist(strsplit(doc, ' '))
  # only words with 2+ characters
  v <- unique(v[grepl('..+', v)])
  # find glove repr
  embed <- lapply(v, FUN = function (x) as.vector(glove[glove$word == x, -1]))
  # create matrix of word vectors (1 row is 1 word)
  m <- matrix(unlist(embed), nrow=length(embed), byrow=TRUE)
  # calculate mean of each dim
  colSums(m) / length(v)
}

loader_glove <- function (dims) {
  # dims - glove dataset dimensions 50/100/200/300
  # Returns train_X, train_y, test_X, test_y in a list
  wine <- load_wines()
  glove <- load_glove(dims)

  all_ids <- wine$X1
  sample_size <- floor(0.8 * nrow(wine))
  train_ids <- sample(all_ids, sample_size)
  test_ids <- setdiff(all_ids, train_ids)
  train <- wine[J(train_ids)]
  test <- wine[J(test_ids)]

  dtm_train_glove <- clean_description(train)
  dtm_train_glove$description <- sapply(dtm_train_glove$description, function (x) embed_doc(x, glove))

  dtm_test_glove <- clean_description(test)
  dtm_test_glove$description <- sapply(dtm_train_glove$description, function (x) embed_doc(x, glove))

  write.csv(dtm_train_glove, 'data/dtm_train_glove.csv')
  write.csv(dtm_test_glove, 'data/dtm_test_glove.csv')

  return (list(dtm_train_glove, train$sentiment,
          dtm_test_glove, test$sentiment))
}
