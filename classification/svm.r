library(e1071)

source("utils/loader_tfidf.r")
source("utils/loader_glove.r")
source("utils/measure_quality.r")

sets_list <- load_glove(oversampling = FALSE)
train_X <- as.matrix(sets_list[[1]][1:15000,])
train_y <- as.vector(sets_list[[2]][1:15000])
test_X <- as.matrix(sets_list[[3]])
test_y <- as.vector(sets_list[[4]])

svm_classifier <- svm(
  train_X,
  factor(train_y),
  kernel = 'radial',
  cost = 1,
  gamma = 1 / ncol(train_X),
  class.weights='inverse'
)
preds <- predict(svm_classifier, test_X)
measure_quality(as.numeric(as.character(preds)), test_y)