library(rpart)

source("utils/loader_tfidf.r")
source("utils/loader_glove.r")
source("utils/measure_quality.r")

sets_list <- load_tfidf()
train_X <- sets_list[[1]]
train_X <- train_X[1:as.integer(0.8*nrow(train_X)), ]
train_y <- sets_list[[2]]
train_y <- train_y[1:as.integer(0.8*length(train_y))]

pca1 <- prcomp(train_X, scale. = FALSE)
train_X <- predict(pca1, train_X)
train_X <- train_X[, 1:30]

valid_X <- sets_list[[1]]
valid_X <- valid_X[as.integer(0.8*nrow(valid_X)+1):nrow(valid_X), ]
valid_y <- sets_list[[2]]
valid_y <- valid_y[as.integer(0.8*length(valid_y)+1):length(valid_y)]
valid_X <- predict(pca1, valid_X)
valid_X <- valid_X[, 1:30]

test_X <- sets_list[[3]]
test_y <- sets_list[[4]]
test_X <- predict(pca1, test_X)
test_X <- test_X[, 1:30]

train <- data.frame(cbind(train_X, train_y))
names(train) <- c(paste('v', 1:ncol(train_X), sep = '_'), 'sentiment')

valid <- data.frame(cbind(valid_X, valid_y))
names(valid) <- c(paste('v', 1:ncol(valid_X), sep = '_'), 'sentiment')

test <- data.frame(cbind(test_y, test_X))
names(test) <- c(paste('v', 1:ncol(test_X), sep = '_'), 'sentiment')

positiveWeight <- nrow(train) / sum(train$sentiment == 1)
negativeWeight <-  nrow(train) / sum(train$sentiment == 0)
modelWeights <- ifelse(train$sentiment== 1, positiveWeight, negativeWeight)

tree_classifier <- rpart(
  sentiment ~ .,
  data = train,
  method = 'class',
  weights = modelWeights,
  parms = list(split = 'gini'), # to be chosen gini vs information
  control = rpart.control(cp = 0.0005, maxdepth = 30)
)

print("############################## TRAIN ##############################")
train_preds <- predict(tree_classifier, train)
train_preds_probs <- apply(train_preds, 1, function (x) {
  if (x[[1]] > x[[2]]) 1 - x[[1]] else x[[2]]
})
measure_quality(train_preds_probs, train_y)

print("############################## VALID ##############################")
valid_preds <- predict(tree_classifier, valid)
valid_preds_probs <- apply(valid_preds, 1, function (x) {
  if (x[[1]] > x[[2]]) 1 - x[[1]] else x[[2]]
})
measure_quality(valid_preds_probs, valid_y)

print("############################## TEST ###############################")
test_preds <- predict(tree_classifier, test)
test_preds_probs <- apply(test_preds, 1, function (x) {
  if (x[[1]] > x[[2]]) 1 - x[[1]] else x[[2]]
})
measure_quality(test_preds_probs, test_y)