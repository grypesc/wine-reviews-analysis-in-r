# Created by: greg
# Created on: 11.11.2020

library(xgboost)
source("utils/loader_tfidf.r")
source("utils/loader_glove.r")
source("utils/measure_quality.r")

sets_list <- load_glove(oversampling = FALSE)
train_X <- sets_list[[1]]
train_y <- sets_list[[2]]
test_X <- sets_list[[3]]
test_y <- sets_list[[4]]

c1 <- sum(train_y)
c0 <- length(train_y) - c1

bst <- xgboost(data = as.matrix(train_X), label = train_y, objective = "binary:logistic",
               max.depth = 10, nrounds = 100,
               scale_pos_weight=as.double(c0)/c1,
              )

print("############################## TRAIN ##############################")
train_preds <- predict(bst, as.matrix(train_X))
measure_quality(train_preds, train_y)

print("############################## TEST ##############################")
test_preds <- predict(bst, as.matrix(test_X))
measure_quality(test_preds, test_y)