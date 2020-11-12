library(rpart)

source("utils/loader_tfidf.r")
source("utils/loader_glove.r")
source("utils/measure_quality.r")

sets_list_glove <- load_glove()
train_X <- sets_list_glove[[1]][1:100,]
train_y <- sets_list_glove[[2]][1:100]
test_X <- sets_list_glove[[3]]
test_y <- sets_list_glove[[4]]

train <- data.frame(cbind(train_X, train_y))

tree_classifier <- rpart(
  train_y ~ .,
  data = train,
  method = 'class',
  parms = list(split = 'information'),
  maxdepth = 10
)

#printcp(tree_classifier) # display the results
#pruned <- prune(tree_classifier, cp=tree_classifier$cptable[which.min(tree_classifier$cptable[,"xerror"]),"CP"])
#printcp(pruned)

preds <- apply(predict(tree_classifier, data.frame(test_X)), 1, function (x) {
  if (x[[1]] > x[[2]]) 0 else 1
})
measure_quality(preds, test_y)