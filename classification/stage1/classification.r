library(text2vec)
library(data.table)
library(readr)

wine <- read_csv('kaggle/input/winemag-data_first150k.csv')
wine <- subset(wine, select = c(X1, points, price, description))
wine$is_positive <- ifelse(wine$points>90, 1, 0)
setDT(wine)
setkey(wine, X1)

# Check missing values in a column
# print(sum(is.na(wine$description)))


set.seed(99)
all_ids <- wine$X1
sample_size <- floor(0.8 * nrow(wine))
train_ids <- sample(all_ids, sample_size)
test_ids <- setdiff(all_ids, train_ids)
train <- wine[J(train_ids)]
test <- wine[J(test_ids)]


# define preprocessing function and tokenization function
prep_fun = tolower
tok_fun = word_tokenizer

it_train = itoken(train$description,
             preprocessor = prep_fun,
             tokenizer = tok_fun,
             ids = train$X1,
             progressbar = TRUE)

vocab = create_vocabulary(it_train)

vectorizer = vocab_vectorizer(vocab)
t1 = Sys.time()
dtm_train = create_dtm(it_train, vectorizer)
print(difftime(Sys.time(), t1, units = 'sec'))

library(glmnet)
t1 = Sys.time()
glmnet_classifier = cv.glmnet(x = dtm_train, y = train[['is_positive']],
                              family = 'binomial',
                              # L1 penalty
                              alpha = 1,
                              # interested in the area under ROC curve
                              type.measure = "auc",
                              # 5-fold cross-validation
                              nfolds = 5,
                              # high value is less accurate, but has faster training
                              thresh = 1e-3,
                              # again lower number of iterations for faster training
                              maxit = 1e3)
print(difftime(Sys.time(), t1, units = 'sec'))

plot(glmnet_classifier)

print(paste("max AUC =", round(max(glmnet_classifier$cvm), 4)))

it_test = tok_fun(prep_fun(test$descritpion))
it_test = itoken(it_test, ids = test$X1,
                 # turn off progressbar because it won't look nice in rmd
                 progressbar = FALSE)

dtm_test = create_dtm(it_test, vectorizer)

preds = predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test$sentiment, preds)