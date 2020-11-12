# Created by: mati
# Created on: 08.11.2020
# Glove pretrained model: http://nlp.stanford.edu/data/glove.6B.zip

library(dplyr)
library(data.table)
library(readr)
library(Matrix)
library(pbapply)
library(hash)

load_wines <- function () {
  wine <- read_csv('data/winemag-data_first150k.csv')
  wine <- subset(wine, select = c(X1, points, price, description))
  wine$sentiment <- ifelse(wine$points>90, 1, 0)
  setDT(wine)
  setkey(wine, X1)
  return(wine[nchar(wine$description) > 0, ])
}

load_glove_model <- function (dims) {
  # dims - glove dataset dimensions 50/100/200/300
  file <- sprintf('data/glove.6B.%dd.txt', dims)
  if (!file.exists(file)) {
    download.file('http://nlp.stanford.edu/data/glove.6B.zip',destfile = 'data/glove.6B.zip')
    unzip('data/glove.6B.zip')
  }
  glove <- fread(file, data.table=F, encoding='UTF-8')
  dict <- hash::hash()
  names(glove) <- c('word', paste('dim', 1:dims, sep = '_'))
  print("creating glove dict")
  for (row in 1:nrow(glove)) {
    word <- glove[row, 'word']
    vec <- glove[row, -1]
    dict[[word]] <- vec
  }
  print("glove dict created")
  return(dict)
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

embed_doc <- function (doc, glove, dims) {
  # find glove repr
  vec <- unlist(strsplit(doc, ' '))
  embeded <- sapply(vec, function (x) {
      if (has.key(x, glove))
        as.numeric(glove[[x]])
      else
        as.numeric(rep(NA, dims))
  })
  return(rowMeans(embeded, na.rm = TRUE))
}

embed_df <- function (df, glove, dims) {
  print("cleaning documents")
  emb <- clean_description(df)
  print("embedding documents")
  emb <- data.frame(t(pbsapply(emb$description, function (x) embed_doc(x, glove, dims))))
  names(emb) <- paste('dim', 1:dims, sep = '_')
  return(data.frame(emb))
}

load_glove_raw <- function (split=0.8, save=FALSE, dims=50) {
  # dims - glove dataset dimensions 50/100/200/300
  # Returns train_X, train_y, test_X, test_y in a list
  wine <- load_wines()
  glove <- load_glove_model(dims)

  all_ids <- wine$X1
  sample_size <- floor(split * nrow(wine))
  train_ids <- sample(all_ids, sample_size)
  test_ids <- setdiff(all_ids, train_ids)
  train <- wine[J(train_ids)]
  test <- wine[J(test_ids)]

  dtm_train_glove <- embed_df(train, glove, dims)
  dtm_test_glove <- embed_df(test, glove, dims)

  if (save) {
    # as data frames
    train_full <- cbind(dtm_train_glove, train$sentiment)
    test_full <- cbind(dtm_test_glove, test$sentiment)
    write_csv(train_full, 'data/train_glove.csv')
    write_csv(test_full, 'data/test_glove.csv')
  }
  train_oversampled <- oversample(dtm_train_glove, train$sentiment)
  train_X <- train_oversampled[[1]]
  train_y <- train_oversampled[[2]]
  test_oversampled <- oversample(dtm_test_glove, test$sentiment)
  test_X <- test_oversampled[[1]]
  test_y <- test_oversampled[[2]]
  return (list(as.matrix(train_X), as.vector(train_y), as.matrix(test_X), as.vector(test_y)))
}

load_glove_from_file <- function() {
  train_full <- read_csv('data/train_glove.csv')
  train_X <- train_full[, 1:ncol(train_full)-1]
  train_y <- train_full[, ncol(train_full)]
  train_oversampled <- oversample(train_X, train_y)
  train_X <- train_oversampled[[1]]
  train_y <- train_oversampled[[2]]

  test_full <- read_csv('data/test_glove.csv')
  test_X <- test_full[, 1:ncol(test_full)-1]
  test_y <- test_full[, ncol(test_full)]
  test_oversampled <- oversample(test_X, test_y)
  test_X <- test_oversampled[[1]]
  test_y <- test_oversampled[[2]]

  return (list(as.matrix(train_X), as.vector(train_y), as.matrix(test_X), as.vector(test_y)))
}

oversample <- function (X, y) {
  full <- cbind(X, y)
  c0 <- sum(full[ncol(full)] == 0)
  c1 <- sum(full[ncol(full)] == 1)
  if (c0 == 0 || c1 == 0) {
    print("there are no enough samples to reproduce")
    return(list(X, y))
  }
  if (c0 > c1) {
    samples <- c0 - c1
    minority <- full[full[ncol(full)] == 1, ]
    sampled <- minority[sample(1:nrow(minority), samples, replace = TRUE), ]
  } else {
    samples <- c1 - c0
    minority <- full[full[ncol(full)] == 0, ]
    sampled <- minority[sample(1:nrow(minority), samples, replace = TRUE), ]
  }
  result <- rbind(full, sampled)
  return(list(result[, 1:ncol(result)-1], result[,ncol(result)]))
}

load_glove <- function () {
  if (file.exists("data/train_glove.csv") && file.exists("data/test_glove.csv")) {
    return (load_glove_from_file())
  }
  return (load_glove_raw(split = 0.8, save = TRUE, dims = 50))
}