library(rpart)

source("utils/loader_tfidf.r")
source("utils/loader_glove.r")
source("utils/measure_quality.r")

sets_list <- load_glove(oversampling = FALSE)
train_X <- as.matrix(sets_list[[1]])
train_y <- as.vector(sets_list[[2]])
test_X <- as.matrix(sets_list[[3]])
test_y <- as.vector(sets_list[[4]])

train <- data.frame(cbind(train_X, train_y))
test <- data.frame(cbind(test_X, test_y))

tree_classifier <- rpart(
  train_y ~ .,
  data = train,
  method = 'class',
  parms = list(split = 'information'), # to be chosen gini vs information
  maxdepth = 10 # to be found
)

pruned <- prune(
  tree_classifier,
  cp=10 # to be found
)


preds <- predict(pruned, subset(test, select=-test_y))
preds <- apply(preds, 1, function (x) {
  if (x[[1]] > x[[2]]) 0 else 1
})
measure_quality(preds, test_y)