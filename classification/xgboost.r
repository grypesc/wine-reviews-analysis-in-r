# Title     : TODO
# Objective : TODO
# Created by: greg
# Created on: 11.11.2020

library(xgboost)
source("utils/loader_tfidf.r")
source("utils/measure_quality.r")

sets_list <- load_tfidf()
train_X <- sets_list[[1]]
train_y <- sets_list[[2]]
test_X <- sets_list[[3]]
test_y <- sets_list[[4]]

bst <- xgboost(data = as.matrix(train_X), label = train_y, max.depth = 4,
               eta = 1, nthread = 2, nrounds = 10,objective = "binary:logistic")

preds <- predict(bst, as.matrix(test_X))
measure_quality(preds, test_y)