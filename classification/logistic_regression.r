library(glmnet)

source("utils/loader_tfidf.r")
source("utils/measure_quality.r")

sets_list <- load_tfidf()
train_X <- sets_list[[1]]
train_y <- sets_list[[2]]
test_X <- sets_list[[3]]
test_y <- sets_list[[4]]

c1 <- sum(train_y)
c0 <- length(train_y) - c1
class_ratio <- c0/c1

t1 <- Sys.time()
glmnet_classifier <- cv.glmnet(x = train_X, y = train_y,
                               family = 'binomial',
                               # L1 penalty
                               alpha = 1,
                               #custom weights to counter class imbalance
                               weights = train_y*0.5 + 1,
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

predictions <- predict(glmnet_classifier, test_X, type = 'response')[, 1]
measure_quality(predictions, test_y, threshold = 0.50)
