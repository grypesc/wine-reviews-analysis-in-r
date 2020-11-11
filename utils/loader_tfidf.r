# Title     : loader_tfidf
# Objective : Load data, turn it into tfidf and return train and test sets
# Created by: greg
# Created on: 10.11.2020
# TODO Paths, oversampling
library(text2vec)
library(data.table)
library(readr)

loader_tfidf <- function() {
  # Returns train_X, train_y, test_X, test_y in a list

  wine <- read_csv('data/winemag-data_first150k.csv')
  wine <- subset(wine, select = c(X1, points, price, description))
  wine$sentiment <- ifelse(wine$points>90, 1, 0)
  setDT(wine)
  setkey(wine, X1)

  all_ids <- wine$X1
  sample_size <- floor(0.8 * nrow(wine))
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

  it_test <- tok_fun(prep_fun(test$description))
  it_test <- itoken(it_test, ids = test$X1,
                    # turn off progressbar because it won't look nice in rmd
                   progressbar = FALSE)
  dtm_test_tfidf <- create_dtm(it_test, vectorizer)
  dtm_test_tfidf <- transform(dtm_test_tfidf, tfidf)
  # Got to return multiple objects in a list because R is bad
  return (list(dtm_train_tfidf, train$sentiment,
          dtm_test_tfidf, test$sentiment))
}