library(randomForest)

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
class_ratio <- c0/c1

t1 <- Sys.time()
forest_classifier <- randomForest(x = as.matrix(train_X), y = as.factor(train_y),
                                  ntree = 100)

print(difftime(Sys.time(), t1, units = 'sec'))
plot(forest_classifier)

print("############################## TRAIN ##############################")
train_preds <- predict(forest_classifier, as.matrix(train_X), type = 'response')
measure_quality(as.numeric(as.character(train_preds)), train_y)

print("############################## TEST ##############################")
test_preds <- predict(forest_classifier, as.matrix(test_X), type = 'response')
measure_quality(as.numeric(as.character(test_preds)), test_y)
