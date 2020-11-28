library(rpart)

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

tree_classifier <- rpart(
  sentiment ~ .,
  data = train,
  method = 'class',
  parms = list(split = 'information'), # to be chosen gini vs information
  maxdepth = 30 # to be found
)

pruned <- prune(
  tree_classifier,
  cp=10 # to be found
)

print("############################## TRAIN ##############################")
train_preds <- predict(tree_classifier, train)
train_preds <- apply(train_preds, 1, function (x) {
  if (x[[1]] > x[[2]]) 1 - x[[1]] else x[[2]]
})
measure_quality(train_preds, train_y)

print("############################## TEST ###############################")
test_preds <- predict(tree_classifier, test)
test_preds <- apply(test_preds, 1, function (x) {
  if (x[[1]] > x[[2]]) 1 - x[[1]] else x[[2]]
})
measure_quality(test_preds, test_y)