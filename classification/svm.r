library(e1071)

source("utils/loader_tfidf.r")
source("utils/loader_glove.r")
source("utils/measure_quality.r")

sets_list <- load_glove()
train_X <- sets_list[[1]]
train_y <- sets_list[[2]]
test_X <- sets_list[[3]]
test_y <- sets_list[[4]]


model <- svm(train_X, train_y)
print(model)
summary(model)
pred <- predict(model, test_X)
measure_quality(pred, matrix(train_y))