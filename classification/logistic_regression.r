library(glmnet)

source("utils/loader_tfidf.r")
source("utils/measure_quality.r")

sets_list <- load_tfidf()
train_X <- sets_list[[1]]
train_y <- sets_list[[2]]
test_X <- sets_list[[3]]
test_y <- sets_list[[4]]

t1 <- Sys.time()
glmnet_classifier = cv.glmnet(x = train_X, y = train_y,
                              family = 'binomial',
                              # L1 penalty
                              alpha = 1,
                              # interested in the area under ROC curve
                              type.measure = "auc",
                              # 5-fold cross-validation
                              nfolds = 5,
                              # high value is less accurate, but has faster training
                              thresh = 1e-2,
                              # again lower number of iterations for faster training
                              maxit = 1e2)
print(difftime(Sys.time(), t1, units = 'sec'))
plot(glmnet_classifier)

preds <- predict(glmnet_classifier, test_X, type = 'response')[, 1]
measure_quality(preds, test_y)
