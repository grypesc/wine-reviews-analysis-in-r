library(randomForest)

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

c1 <- sum(train_y)
c0 <- length(train_y) - c1
class_ratio <- c0/c1

t1 <- Sys.time()
forest_classifier <- randomForest(x = as.matrix(train_X), y = as.factor(train_y),
                                  ntree = 101, mtry = 10)

print(difftime(Sys.time(), t1, units = 'sec'))
plot(forest_classifier)

print("############################## TRAIN ##############################")
train_preds <- predict(forest_classifier, as.matrix(train_X), type = 'prob')[, 2]
measure_quality(as.numeric(as.character(train_preds)), train_y)

print("############################## VALID ##############################")
valid_preds <- predict(forest_classifier, as.matrix(valid_X), type = 'prob')[, 2]
measure_quality(as.numeric(as.character(valid_preds)), valid_y)


#print("############################## TEST ##############################")
#test_preds <- predict(forest_classifier, as.matrix(test_X), type = 'response')
#measure_quality(as.numeric(as.character(test_preds)), test_y)
