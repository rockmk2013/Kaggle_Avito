library(tidyverse)
library(lubridate)
library(magrittr)
library(text2vec)
library(tokenizers)
library(stopwords)
library(xgboost)
library(Matrix)
library(stringr)
library(stringi)
library(forcats)
set.seed(0)

#---------------------------
cat("Loading data...\n")
tr <- read_csv("input/train.csv") 
te <- read_csv("input/test.csv")

#---------------------------
cat("Preprocessing...\n")
tri <- 1:nrow(tr)
y <- tr$deal_probability

#挑選除了目標外的變數，並新增欄位
tr_te <- tr %>% 
  select(-deal_probability) %>% 
  bind_rows(te) %>% 
  mutate(no_img = is.na(image) %>% as.integer(),
         no_dsc = is.na(description) %>% as.integer(),
         no_p1 = is.na(param_1) %>% as.integer(), 
         no_p2 = is.na(param_2) %>% as.integer(), 
         no_p3 = is.na(param_3) %>% as.integer(),
         titl_len = str_length(title),
         desc_len = str_length(description),
         titl_capE = str_count(title, "[A-Z]"),
         titl_capR = str_count(title, "[А-Я]"),
         desc_capE = str_count(description, "[A-Z]"),
         desc_capR = str_count(description, "[А-Я]"),
         titl_cap = str_count(description, "[A-ZА-Я]"),
         desc_cap = str_count(description, "[A-ZА-Я]"),
         titl_pun = str_count(title, "[[:punct:]]"),
         desc_pun = str_count(description, "[[:punct:]]"),
         titl_dig = str_count(title, "[[:digit:]]"),
         desc_dig = str_count(description, "[[:digit:]]"),
         user_type = factor(user_type),
         category_name = factor(category_name) %>% as.integer(),
         parent_category_name = factor(parent_category_name) %>% as.integer(), 
         region = factor(region) %>% as.integer(),
         param_1 = factor(param_1) %>% as.integer(),
         param_2 = factor(param_2) %>% as.integer(),
         param_3 = factor(param_3) %>% fct_lump(prop = 0.00005) %>% as.integer(),
         city =  factor(city) %>% fct_lump(prop = 0.0003) %>% as.integer(),
         user_id = factor(user_id) %>% fct_lump(prop = 0.000025) %>% as.integer(),
         price = log1p(price),
         txt = paste(title, description, sep = " "),
         mday = mday(activation_date),
         wday = wday(activation_date),         
         day = day(activation_date)) %>% 
  select(-item_id, -image, -title, -description, -activation_date) %>% 
  replace_na(list(image_top_1 = -1, price = -1, 
                  param_1 = -1, param_2 = -1, param_3 = -1, titl_cap = 0,
                  desc_len = 0, desc_cap = 0, desc_pun = 0, desc_dig = 0,
                  desc_capE = 0, desc_capR = 0, desc_lowE = 0, desc_lowR = 0)) %T>% 
  glimpse()

rm(tr, te); gc()

#---------------------------
cat("Parsing text...\n")
it <- tr_te %$%
  str_to_lower(txt) %>%
  str_replace_all("[^[:alpha:]]", " ") %>%
  str_replace_all("\\s+", " ") %>%
  tokenize_word_stems(language = "russian") %>% 
  itoken()

vect <- create_vocabulary(it, ngram = c(1, 1), stopwords = stopwords("ru")) %>%
  prune_vocabulary(term_count_min = 3, doc_proportion_max = 0.4, vocab_term_max = 12500) %>% 
  vocab_vectorizer()

m_tfidf <- TfIdf$new(norm = "l2", sublinear_tf = T)
tfidf <-  create_dtm(it, vect) %>% 
  fit_transform(m_tfidf)

rm(it, vect, m_tfidf); gc()

#---------------------------
cat("Preparing data...\n")
X <- tr_te %>% 
  select(-txt) %>% 
  sparse.model.matrix(~ . - 1, .) %>% 
  cbind(tfidf)
A =  tr_te %>% 
  select(-txt) %>% 
  sparse.model.matrix(~ . - 1, .)
rm(tr_te, tfidf); gc()

dtest <- xgb.DMatrix(data = X[-tri, ])
X <- X[tri, ]; gc()
tri <- caret::createDataPartition(y, p = 0.9, list = F) %>% c()
dtrain <- xgb.DMatrix(data = X[tri, ], label = y[tri])
dval <- xgb.DMatrix(data = X[-tri, ], label = y[-tri])
cols <- colnames(X)

rm(X, y, tri); gc()

#-------------first model--------------
cat("Training model...\n")
p <- list(objective = "reg:logistic",
          booster = "gbtree",
          eval_metric = "rmse",
          nthread = 8, #cpu 數量
          eta = 0.05,
          max_depth = 17, 
          min_child_weight = 10,
          gamma = 0,
          subsample = 0.8,
          colsample_bytree = 0.7,
          alpha = 2,
          lambda = 0)

#m_xgb <- xgb.train(p, dtrain, p$nrounds, list(val = dval), print_every_n = 10, early_stopping_rounds = 50)

#---------------------------
xgbcv <- xgb.cv( params = p, data = dtrain, nrounds = 100, nfold = 5, showsd = T, stratified = T, print.every.n = 10, early.stop.round = 20, maximize = F)
n_iter = min(xgbcv$test.error.mean)
xgb1 <- xgb.train (params = p, data = dtrain, nrounds = n_iter, watchlist = list(val=dval,train=dtrain), print.every.n = 10, early.stop.round = 10, maximize = F)
#---------------------------
#2.1 Tune max_depth and min_child_weight
# XGBoost
# Set control
xgb_trcontrol_1 = trainControl(method = "cv", number = 3, allowParallel = TRUE)

# Set grid and train
xgb_grid_1 <- expand.grid(nrounds=2000, eta = 0.05, 
                          gamma = 0, max_depth = c(15,17,19,21), 
                          min_child_weight = c(7,9,11,13,15), subsample=0.8, 
                          colsample_bytree=0.7)

xgb_train_1 <- train(x = trainMatrix, y = as.factor(train_ans), trControl = xgb_trcontrol_1,tuneGrid = xgb_grid_1, method = "xgbTree")

# Keep the best results for second round tuning
max_d <- xgb_train_1$bestTune$max_depth
min_c <- xgb_train_1$bestTune$min_child_weight

# Second round tuning
xgb_grid_2 <- expand.grid(nrounds=best_iter, eta = 0.1, gamma = 0, max_depth = max_d, min_child_weight = c(min_c-0.5,min_c,min_c+0.5), subsample=0.8, colsample_bytree=0.7)
xgb_train_2 <- train(x = trainMatrix, y = as.factor(train_ans), trControl = xgb_trcontrol_1, tuneGrid = xgb_grid_2, method = "xgbTree")
min_c <- xgb_train_2$bestTune$min_child_weight


#2.2 Tune gamma (best at gamma = 0)
xgb_grid_3 <- expand.grid(nrounds=best_iter, eta = 0.1, gamma = c(seq(from=0,to=1,by=0.1)), max_depth = max_d, min_child_weight = min_c, subsample=0.8, colsample_bytree=0.7)
xgb_train_3 = train(x = trainMatrix, y = as.factor(train_ans), trControl = xgb_trcontrol_1, tuneGrid = xgb_grid_3, method = "xgbTree")
gamma <- xgb_train_3$bestTune$gamma


#2.3 Recalibrate number of boosting rounds
param_2 <- list(booster = "gbtree", objective = "binary:logistic", eval_metric = "error",  eta = 0.1, gamma = gamma, max_depth = max_d, min_child_weight = min_c, subsample=0.8, colsample_bytree=0.7)
xgb_meta_cv_2 <- xgb.cv( params = param_2, data = trainMatrix, nrounds = 1000, nfold = 5, label = train_ans, showsd = T, stratified = T, early_stopping_rounds = 50, maximize = F, print_every_n = 5)
best_iter_2 <- xgb_meta_cv_2$best_iteration

#2.4 Tune subsample and column sample by tree (subsample = 0.8, colsample_bytree = 0.7)
xgb_grid_4 <- expand.grid(nrounds=best_iter_2, eta = 0.1, gamma = gamma, max_depth = max_d, min_child_weight = min_c, subsample = c(0.6,0.7,0.8,0.9,1), colsample_bytree = c(0.6,0.7,0.8,0.9,1))
xgb_train_4 <- train(x = trainMatrix, y = as.factor(train_ans), trControl = xgb_trcontrol_1, tuneGrid = xgb_grid_4, method = "xgbTree")
subsample <- xgb_train_4$bestTune$subsample
colsample_bytree <- xgb_train_4$bestTune$colsample_bytree

xgb_grid_5 <- expand.grid(nrounds=best_iter_2, eta = 0.1, gamma = gamma, max_depth = max_d, min_child_weight = min_c, subsample=c(subsample-0.05,subsample,subsample+0.5), colsample_bytree=c(colsample_bytree-0.05,colsample_bytree,colsample_bytree+0.05))

xgb_train_5 <- train(x = trainMatrix, y = as.factor(train_ans), trControl = xgb_trcontrol_1, tuneGrid = xgb_grid_5, method = "xgbTree")

subsample <- xgb_train_5$bestTune$subsample
colsample_bytree <- xgb_train_5$bestTune$colsample_bytree

#3 Tune nrounds for smaller eta one last time (n = 1600)
param_3 <- list(booster = "gbtree", objective = "binary:logistic", eval_metric = "error",  eta = 0.01, gamma = gamma, max_depth = max_d, min_child_weight = min_c, subsample=subsample, colsample_bytree=colsample_bytree)
xgb_meta_cv_3 <- xgb.cv( params = param_3, data = trainMatrix, nrounds = 1600, nfold = 5, label = train_ans, showsd = T, stratified = T, early_stopping_rounds = 50, maximize = F, print_every_n = 5)
best_iter_final <- xgb_meta_cv_3$best_iteration
#---------------------------
xgb.importance(cols, model = m_xgb) %>%   
  xgb.plot.importance(top_n = 35)

cat("Creating submission file...\n")
read_csv("input/sample_submission.csv")  %>%  
  mutate(deal_probability = predict(m_xgb, dtest)) %>%
  write_csv(paste0("xgb_tfidf", m_xgb$best_score, ".csv"))