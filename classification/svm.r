library(e1071)

source("utils/loader_tfidf.r")
source("utils/loader_glove.r")
source("utils/measure_quality.r")

sets_list <- load_glove(oversampling = FALSE)
train_X <- sets_list[[1]]
train_X <- train_X[1:as.integer(0.8*nrow(train_X)), ]
train_y <- sets_list[[2]]
train_y <- train_y[1:as.integer(0.8*length(train_y))]

positive <- train_y == 1
negative <- train_y == 0
train_X <- rbind(train_X[positive, ][1:5000, ], train_X[negative, ][1:5000, ])
train_y <- c(train_y[positive][1:5000], train_y[negative][1:5000])

valid_X <- sets_list[[1]]
valid_X <- valid_X[as.integer(0.8*nrow(valid_X)+1):nrow(valid_X), ]
valid_y <- sets_list[[2]]
valid_y <- valid_y[as.integer(0.8*length(valid_y)+1):length(valid_y)]

positive <- valid_y == 1
negative <- valid_y == 0
valid_X <- rbind(valid_X[positive, ][1:5000, ], valid_X[negative, ][1:5000, ])
valid_y <- c(valid_y[positive][1:5000], valid_y[negative][1:5000])

test_X <- sets_list[[3]]
test_y <- sets_list[[4]]

svm_classifier <- svm(
  train_X,
  as.factor(train_y),
  type = 'C-classification',
  kernel = 'radial',
  probability = TRUE,
  cost=50,
  gamma = 0.01/ncol(train_X)
)

print("############################## TRAIN ##############################")
train_preds <- predict(svm_classifier, train_X, probability = TRUE)
train_preds <- apply(attr(train_preds, "probabilities"), 1, function (x) {
  if (x[[1]] > x[[2]]) 1 - x[[1]] else x[[2]]
})
measure_quality(as.numeric(as.character(train_preds)), train_y)

print("############################## TEST ###############################")
test_preds <- predict(svm_classifier, valid_X, probability = TRUE)
test_preds <- apply(attr(test_preds, "probabilities"), 1, function (x) {
  if (x[[1]] > x[[2]]) 1 - x[[1]] else x[[2]]
})
measure_quality(as.numeric(as.character(test_preds)), valid_y)