# Objective : Training and evaluating SVM models
# Created by: matkob
# Created on: 15.11.2020

library(e1071)

source("utils/loader_tfidf.r")
source("utils/loader_glove.r")
source("utils/measure_quality.r")

sets_list <- load_glove(oversampling = FALSE)
train_X <- sets_list[[1]]
train_X <- train_X[1:as.integer(0.8*nrow(train_X)), ]
train_y <- sets_list[[2]]
train_y <- train_y[1:as.integer(0.8*length(train_y))]

valid_X <- sets_list[[1]]
valid_X <- valid_X[as.integer(0.8*nrow(valid_X)+1):nrow(valid_X), ]
valid_y <- sets_list[[2]]
valid_y <- valid_y[as.integer(0.8*length(valid_y)+1):length(valid_y)]

test_X <- sets_list[[3]]
test_y <- sets_list[[4]]

svm_classifier <- svm(
  train_X,
  as.factor(train_y),
  type = 'C-classification',
  # testing just radial kernel because of SVM's high complexity
  kernel = 'radial',
  # return probabilities of classes
  probability = TRUE,
  # cost of breaking the margin, to be tuned
  cost=1,
  # relevance of distant samples, to be tuned
  gamma = 1/ncol(train_X),
  # just a speed tweak
  cachesize = 8192,
)

print("############################## TRAIN ##############################")
train_preds <- predict(svm_classifier, train_X, probability = TRUE)
train_preds_probs <- apply(attr(train_preds, "probabilities"), 1, function (x) {
  # model output is prob(0), prob(1), selecting just the probability of 1
  if (x[[1]] > x[[2]]) 1 - x[[1]] else x[[2]]
})
measure_quality(as.numeric(as.character(train_preds_probs)), train_y)

print("############################## VALID ##############################")
valid_preds <- predict(svm_classifier, valid_X, probability = TRUE)
valid_preds_probs <- apply(attr(valid_preds, "probabilities"), 1, function (x) {
  if (x[[1]] > x[[2]]) 1 - x[[1]] else x[[2]]
})
measure_quality(as.numeric(as.character(valid_preds_probs)), valid_y)

print("############################## TEST ###############################")
test_preds <- predict(svm_classifier, test_X, probability = TRUE)
test_preds_probs <- apply(attr(test_preds, "probabilities"), 1, function (x) {
  if (x[[1]] > x[[2]]) 1 - x[[1]] else x[[2]]
})
measure_quality(as.numeric(as.character(test_preds_probs)), test_y)