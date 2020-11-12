# Title     : loader_tfidf
# Objective : Preprocess data and return train and test sets
# Created by: greg
# Created on: 10.11.2020
# TODO Paths, oversampling

library(text2vec)
library(data.table)
library(readr)
library(Matrix)

load_tfidf <- function () {
  if (file.exists("data/train_tfidf.mm") && file.exists("data/test_tfidf.mm")) {
    return (load_tfidf_from_file())
  }
  return (load_tfidf_raw(split = 0.8, save = TRUE))
}

load_tfidf_raw <- function(split=0.8, save=FALSE) {
  # Load raw data, turn it into tfidf and return train and test sets
  # Returns train_X, train_y, test_X, test_y in a list
  # split - fraction of whole dataset that will be used as training set, the rest is test set
  # save - if write train and test matrices to files
  wine <- read_csv('data/winemag-data_first150k.csv')
  wine <- subset(wine, select = c(X1, points, price, description))
  wine$sentiment <- ifelse(wine$points>90, 1, 0)

  setDT(wine)
  setkey(wine, X1)

  all_ids <- wine$X1
  sample_size <- floor(split * nrow(wine))
  train_ids <- sample(all_ids, sample_size)
  test_ids <- setdiff(all_ids, train_ids)
  train <- wine[J(train_ids)]
  test <- wine[J(test_ids)]

  # define preprocessing function and tokenization function
  prep_fun <- tolower
  tok_fun <- word_tokenizer

  it_train <- itoken(train$description,
                     preprocessor = prep_fun,
                     tokenizer = tok_fun,
                     ids = train$X1,
                     progressbar = FALSE)

  stop_words <- readLines("utils/eng_stop_words.txt")
  vocab <- create_vocabulary(it_train, stopwords = stop_words)
  pruned_vocab <- prune_vocabulary(vocab,
                                   term_count_min = 10,
                                   doc_proportion_max = 0.5,
                                   doc_proportion_min = 0.001)
  vectorizer <- vocab_vectorizer(pruned_vocab)
  # create document term matrix
  dtm_train <- create_dtm(it_train, vectorizer)
  # define tfidf model
  tfidf <- TfIdf$new()
  # fit model to train data and transform train data with fitted model
  dtm_train_tfidf <- fit_transform(dtm_train, tfidf)
  # tfidf modified by fit_transform() call!
  # apply pre-trained tf-idf transformation to test data
  it_test <- itoken(test$description,
                    preprocessor = prep_fun,
                    tokenizer = tok_fun,
                    ids = test$X1,
                    # turn off progressbar because it won't look nice in rmd
                    progressbar = FALSE)
  dtm_test_tfidf <- create_dtm(it_test, vectorizer)
  dtm_test_tfidf <- transform(dtm_test_tfidf, tfidf)

  #oversampled_train <- oversample(dtm_train_tfidf, train$sentiment)
  #oversampled_train_X <- oversampled_train[[1]]
  #oversampled_train_y <- oversampled_train[[2]]

  if (save) {
    # writing as sparse matrices
    train_full <- cbind(dtm_train_tfidf, train$sentiment)
    test_full <- cbind(dtm_test_tfidf, test$sentiment)
    writeMM(train_full, 'data/train_tfidf.mm')
    writeMM(test_full, 'data/test_tfidf.mm')
  }

  return (list(dtm_train_tfidf, train$sentiment,
          dtm_test_tfidf, test$sentiment))
}

load_tfidf_from_file <- function() {
  train_full <- readMM('data/train_tfidf.mm')
  train_X <- train_full[, 1:ncol(train_full)-1]
  train_y <- train_full[, ncol(train_full)]

  test_full <- readMM('data/test_tfidf.mm')
  test_X <- test_full[, 1:ncol(test_full)-1]
  test_y <- test_full[, ncol(test_full)]

  return (list(train_X, train_y,
          test_X, test_y))
}

oversample <- function (X, y) {
  full <- cbind(X, y)
  c0 <- nrow(full[full[,ncol(full)] == 0,])
  c1 <- nrow(full[full[,ncol(full)] == 1,])
  if (c0 == 0 || c1 == 0) {
    print("there are no enough samples to reproduce")
    return(list(X, y))
  }
  if (c0 > c1) {
    remaining_samples <- c0 - c1
    minority <- full[full[,ncol(full)] == 1, ]
  } else {
    remaining_samples <- c1 - c0
    minority <- full[full[,ncol(full),] == 0, ]
  }
  repeat {
    new_samples <- minority[1:min(remaining_samples, nrow(minority)),]
    full <- rbind(full, new_samples)
    remaining_samples <- remaining_samples - nrow(new_samples)
    if (remaining_samples <= 0) {
      break
    }
  }

  random_rows <- sample(nrow(full))
  result <- full[random_rows,]
  return(list(result[, 1:ncol(result)-1], result[,ncol(result)]))
}