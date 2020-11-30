# Objective : Training and evaluating xgboost models
# Created by: greg
# Created on: 15.11.2020

library(xgboost)
source("utils/loader_tfidf.r")
source("utils/loader_glove.r")
source("utils/measure_quality.r")

sets_list <- load_tfidf()
train_X <- sets_list[[1]]
train_X <- train_X[1:as.integer(0.8*nrow(train_X)), ]
train_y <- sets_list[[2]]
train_y <- train_y[1:as.integer(0.8*length(train_y))]

valid_X <- sets_list[[1]]
valid_X <- valid_X[as.integer(0.8*nrow(valid_X)+1):nrow(valid_X), ]
valid_y <- sets_list[[2]]
valid_y <- valid_y[as.integer(0.8*length(valid_y)+1):length(valid_y)]

full_train_X <- sets_list[[1]]
full_train_y <- sets_list[[2]]

test_X <- sets_list[[3]]
test_y <- sets_list[[4]]

c1 <- sum(train_y)
c0 <- length(train_y) - c1

bst <- xgboost(data = as.matrix(train_X), label = train_y, objective = "binary:logistic",
               max.depth = 15, nrounds = 101, subsample=0.5, nthread = 4,
               scale_pos_weight=as.double(c0)/c1
              )

print("############################## TRAIN ##############################")
train_preds <- predict(bst, as.matrix(train_X))
measure_quality(train_preds, train_y)

print("############################## VALID ##############################")
valid_preds <- predict(bst, as.matrix(valid_X))
measure_quality(valid_preds, valid_y)

#print("############################## TEST ##############################")
#test_preds <- predict(bst, as.matrix(test_X))
#measure_quality(test_preds, test_y)