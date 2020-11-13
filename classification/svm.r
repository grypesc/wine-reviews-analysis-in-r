library(e1071)

source("utils/loader_tfidf.r")
source("utils/loader_glove.r")
source("utils/measure_quality.r")

sets_list <- load_glove(oversampling = FALSE)
train_X <- as.matrix(sets_list[[1]])
train_y <- as.vector(sets_list[[2]])
test_X <- as.matrix(sets_list[[3]])
test_y <- as.vector(sets_list[[4]])

train <- data.frame(cbind(train_y, train_X))
names(train) <- c('sentiment', paste('v', 1:ncol(train_X), sep = '_'))
test <- data.frame(cbind(test_y, test_X))
names(test) <- c('sentiment', paste('v', 1:ncol(test_X), sep = '_'))

svm_classifier <- svm(
  sentiment ~ .,
  data = train,
  kernel = 'linear',
  cost = 10,
  gamma = 1,
  class.weights = 'inverse',
  type = 'C-classification',
  probability = TRUE
)
preds <- predict(svm_classifier, test, probability = TRUE)
preds <- apply(attr(preds, "probabilities"), 1, function (x) {
  if (x[[1]] > x[[2]]) 1 - x[[1]] else x[[2]]
})
measure_quality(as.numeric(as.character(preds)), test_y)