library(e1071)

source("utils/loader_tfidf.r")
source("utils/loader_glove.r")
source("utils/measure_quality.r")

sets_list_glove <- load_glove()
train_X <- sets_list_glove[[1]]
train_y <- sets_list_glove[[2]]
test_X <- sets_list_glove[[3]]
test_y <- sets_list_glove[[4]]

svm_classifier <- svm(train_X, train_y, kernel = 'radial', cost = 1, gamma = 1 / ncol(train_X))
preds <- predict(svm_classifier, test_X)
measure_quality(preds, test_y)