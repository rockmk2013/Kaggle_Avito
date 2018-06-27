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
library(caret)
library(ggplot2)
library(corrplot)
library(mlr)
library(e1071)
set.seed(0)

#read data and preprocessing 
#---------------------------
cat("Loading data...\n")
tr <- read_csv("input/train.csv") 
te <- read_csv("input/test.csv")

#---------------------------
cat("Preprocessing...\n")
tri <- 1:nrow(tr)
y <- tr$deal_probability

#price
pricena_tr = which(is.na(tr$price))
pricena_te = which(is.na(te$price))
#補平均值
tr$price[pricena_tr] = mean(tr$price,na.rm=TRUE)
te$price[pricena_te] = mean(te$price,na.rm=TRUE)

#add feature
#挑選除了目標外的變數，並新增欄位
tr_te <- tr %>% 
  select(-deal_probability) %>% 
  bind_rows(te) %>% 
  mutate(no_img = is.na(image) %>% as.integer(), #image na proceesing
         no_dsc = is.na(description) %>% as.integer(), #description na processing
         no_p1 = is.na(param_1) %>% as.integer(), #param1 na processing
         no_p2 = is.na(param_2) %>% as.integer(), #param2 na processing
         no_p3 = is.na(param_3) %>% as.integer(), #param3 na processing
         titl_len = str_length(title), #長度
         desc_len = str_length(description), #長度
         titl_capE = str_count(title, "[A-Z]"), #標題英文
         titl_capR = str_count(title, "[А-Я]"), #標題俄文
         desc_capE = str_count(description, "[A-Z]"), #敘述英文
         desc_capR = str_count(description, "[А-Я]"), #敘述俄文
         titl_cap = str_count(description, "[A-ZА-Я]"), #混合標題
         desc_cap = str_count(description, "[A-ZА-Я]"), #混合敘述
         titl_pun = str_count(title, "[[:punct:]]"), #標點
         desc_pun = str_count(description, "[[:punct:]]"), #標點符號
         titl_dig = str_count(title, "[[:digit:]]"), #位元
         desc_dig = str_count(description, "[[:digit:]]"), #位元
         user_type = factor(user_type)  ,
         category_name = factor(category_name) %>% as.integer(),
         parent_category_name = factor(parent_category_name) %>% as.integer(), 
         region = factor(region) %>% as.integer(),
         param_1 = factor(param_1) %>% as.integer(),
         param_2 = factor(param_2) %>% as.integer(),
         param_3 = factor(param_3) %>% fct_lump(prop = 0.00005) %>% as.integer(),
         city =  factor(city) %>% fct_lump(prop = 0.0003) %>% as.integer(),
         user_id = factor(user_id) %>% fct_lump(prop = 0.000025) %>% as.integer(),
         price = log(price+0.001),
         txt = paste(title, description, sep = " "),
         mday = mday(activation_date),
         wday = wday(activation_date),
         isweek = if_else(wday<6,1,0)) %>% 
  select(-title, -description, -activation_date) %>% 
  replace_na(list(image_top_1 = -1,  #剩餘變數空值的處理
                  param_1 = -1, param_2 = -1, param_3 = -1, titl_cap = 0,
                  desc_len = 0, desc_cap = 0, desc_pun = 0, desc_dig = 0,
                  desc_capE = 0, desc_capR = 0, desc_lowE = 0, desc_lowR = 0)) %T>% 
  glimpse()

#-----memory manage-----


#-----------read svd data----------------
cat("Loading svd data...\n")
svd_train <- read_delim("input/train_feature.csv","\t", escape_double = FALSE, trim_ws = TRUE) 
svd_test <- read_delim("input/test_feature.csv","\t", escape_double = FALSE, trim_ws = TRUE)
svd_all <- rbind(svd_train,svd_test)
# bind svd feature to training data 
svd_all <- svd_all[,c(2,18:36)]

#-----------read stat data----------------
cat("Loading stat data...\n")
stat_train <- read_csv("input/train_feature_mean_sd.csv") 
stat_test <- read_csv("input/test_feature_mean_sd.csv")
tr <- tr %>% replace_na(list(image_top_1 = -1))
a = tr %>% group_by(category_name) %>% 
  summarise(category_price_skewness = skewness(price,na.rm = TRUE),
            category_deal_prob_skewness = skewness(deal_probability,na.rm = TRUE))
b = tr %>% group_by(image_top_1) %>% 
  summarise(image_top_1_price_skewness = skewness(price,na.rm = TRUE),
            image_top_1_deal_prob_skewness = skewness(deal_probability,na.rm = TRUE))
c = tr %>% group_by(category_name) %>% 
  summarise(category_price_kurtosis = kurtosis(price,na.rm = TRUE),
            category_deal_prob_kurtosis = kurtosis(deal_probability,na.rm = TRUE))
d = tr %>% group_by(image_top_1) %>% 
  summarise(image_top_1_price_kurtosis = kurtosis(price,na.rm = TRUE),
            image_top_1_deal_prob_kurtosis = kurtosis(deal_probability,na.rm = TRUE))
category_stat <- cbind(a,c[,2:3])
image_stat <- cbind(b,d[,2:3])

stat_all <- rbind(stat_train[,-1],stat_test[,-1])
stat_skew <- tr %>% select(1,6,17) %>% 
  left_join(category_stat,by = c("category_name"="category_name")) 
stat_skew <- stat_skew %>% left_join(image_stat,by = c("image_top_1"="image_top_1")) 

svd_all <- svd_all %>% left_join(stat_all,by = c("item_id"="item_id"))
svd_all <- svd_all %>% left_join(stat_skew[,c(1,4:11)],by = c("item_id" = "item_id"))
rm(stat_all,stat_train,stat_test,stat_skew,a,b,c,d)
#-----------read Image data---------------
image_train <- read_csv("input/trainimagefeature.csv")
image_test <- read_csv("input/testimagefeature.csv")
image_bind <- rbind(image_train,image_test)

#-----------read Image data---------------
image_quality_1 <- read_csv("input/image_quality/features/_.csv")
image_quality_2 <- read_csv("input/image_quality/features/test.csv")
image_quality_3 <- read_csv("input/image_quality/features/train-0.csv")
image_quality_4 <- read_csv("input/image_quality/features/train-1.csv")
image_quality_5 <- read_csv("input/image_quality/features/train-2.csv")
image_quality_6 <- read_csv("input/image_quality/features/train-3.csv")
image_quality_7 <- read_csv("input/image_quality/features/train-4.csv")
image_quality_all <- rbind(image_quality_1,image_quality_2,image_quality_3,image_quality_4,
                           image_quality_5,image_quality_6,image_quality_7)
#刪除重複的column
image_quality_all <- image_quality_all %>% distinct(image, .keep_all = T)
rm(image_quality_1,image_quality_2,image_quality_3,image_quality_4,
   image_quality_5,image_quality_6,image_quality_7);gc()
rm(tr, te); gc()
#-----------------kmeans feature------------------------
#SVD分群
km11 <- kmeans(svd_all[,c(6:20)],centers=11,nstart=10)
wss11=km11$tot.withinss
bss11=km11$betweenss
wss11/bss11
clust <- km11$cluster

#image quality 分群
km11_image <- kmeans(image_quality_all[,c(2:14)],centers=11,nstart=10)
wss11_image=km11_image$tot.withinss
bss11_image=km11_image$betweenss
wss11_image/bss11_image
clust_image <- km11_image$cluster
image_quality_all <- cbind(image_quality_all,clust_image)

summarzie_image <- image_quality_all %>% select(-image) %>% group_by(clust_image) %>% summarise_all(funs(mean=mean))

#----------------ridge feature-------------------------
ridge <- read_csv("input/ridge.csv")

#-----------train bind--------------------
tr_te$image <- paste0(tr_te$image,".jpg")

train_bind <- tr_te %>% left_join(svd_all,by=c("item_id"="item_id"))
train_bind <- train_bind %>% left_join(image_bind,by=c("item_id"="item_id"))
train_bind <- cbind(train_bind,clust)
train_bind <- train_bind %>% left_join(ridge,by=c("item_id"="item_id"))
#2011862 -> 2011965 (有重複的Col)
train_bind <- train_bind %>% left_join(image_quality_all, by = c("image"="image"))
train_bind <- train_bind %>% select(-item_id,-image)#把item id,image 拿掉




#------------deal null value-------------
find_na <- function(x){(which(is.na(x)))}
train_bind[1807048,c(36,37,55,56)] <- train_bind[1807047,c(36,37,55,56)] #NA
train_bind$pred1[which(is.na(train_bind$pred1))] <- "NONE"
train_bind$pred2[which(is.na(train_bind$pred2))] <- "NONE"
train_bind$pred3[which(is.na(train_bind$pred3))] <- "NONE"
#image quality none
for (i in 76:89) {
  train_bind[,i][which(is.na(train_bind[,i]))] <- -1 
}
#stat na processing
for (i in 53:68) {
  train_bind[,i][which(is.na(train_bind[,i]))] = mean(train_bind[,i],na.rm=TRUE)
}
#----------deal infinite and factor----------
train_bind$clust_image <- as.factor(train_bind$clust_image)
train_bind$title_desc_len_ratio[which(is.infinite(train_bind$title_desc_len_ratio))] <- -1
which(is.na(train_bind))
str(train_bind)

rm(svd_all,svd_train,svd_test,image_bind,image_test,image_train,km11,ridge)
gc()
#TFIDF feature
#---------------------------
cat("Parsing text...\n")
it <- train_bind %$%
  str_to_lower(txt) %>%
  str_replace_all("[^[:alpha:]]", " ") %>%
  str_replace_all("\\s+", " ") %>%
  tokenize_word_stems(language = "russian") %>% 
  itoken()

vect <- create_vocabulary(it, ngram = c(1, 1), stopwords = stopwords("ru")) %>%
  prune_vocabulary(term_count_min = 3, doc_proportion_max = 0.4, vocab_term_max = 13000) %>% 
  vocab_vectorizer()

m_tfidf <- TfIdf$new(norm = "l2", sublinear_tf = T)
tfidf <-  create_dtm(it, vect) %>% 
  fit_transform(m_tfidf)

#-----memory manage-----
rm(it, vect, m_tfidf); gc()


#----add tune code------
# tr_te <- tr_te[-((length(y)+1):nrow(tr_te)),] %>% dplyr::select(-item_id, -txt)
# train <- tr_te # copying does not change memory usage
#train$deal_probability <- y

tune_train <- train_bind[tri,] %>% dplyr::select(-txt,-pred1,-pred2,-pred3) 
tune_train$deal_probability <- y

#-----reduction city----
a= tune_train %>% group_by(city) %>% summarise(mean = mean(deal_probability)) %>% mutate(city_reduction = if_else(mean<0.13,0,if_else(mean<0.16,1,2))) 
city_reduction <- a[,c(1,3)]

train_bind <- train_bind %>% left_join(city_reduction,by = c("city"="city"))

#-----reduction category----
a= tune_train %>% group_by(category_name) %>% summarise(mean = mean(deal_probability)) %>% mutate(category_reduction = if_else(mean<0.13,0,if_else(mean<0.18,1,2))) 
category_reduction <- a[,c(1,3)]

train_bind <- train_bind %>% left_join(category_reduction,by = c("category_name","category_name"))

#-----reduction param1----
a= tune_train %>% group_by(param_1) %>% summarise(mean = mean(deal_probability)) %>% mutate(param1_reduction = if_else(mean<0.2,0,1)) 
param1_reduction <- a[,c(1,3)]

train_bind <- train_bind %>% left_join(param1_reduction,by = c("param_1","param_1"))

#-----reduction param2----
a= tune_train %>% group_by(param_2) %>% summarise(mean = mean(deal_probability)) %>% mutate(param2_reduction = if_else(mean<0.2,0,1)) 
param2_reduction <- a[,c(1,3)]

train_bind <- train_bind %>% left_join(param2_reduction,by = c("param_2","param_2"))

train_bind$param2_reduction[param2_none] <- sample(c(0,1),replace = TRUE)param2_none <- which(is.na(train_bind$param2_reduction))

#-----reduction param3----
a= tune_train %>% group_by(param_3) %>% summarise(mean = mean(deal_probability)) %>% mutate(param3_reduction = if_else(mean<0.2,0,1)) 
param3_reduction <- a[,c(1,3)]

train_bind <- train_bind %>% left_join(param3_reduction,by = c("param_3","param_3"))

which(is.na(train_bind))
# tune_train$pred1 <- as.integer(tune_train$pred1)
# tune_train$pred2 <- as.integer(tune_train$pred2)
# tune_train$pred3 <- as.integer(tune_train$pred3)

# select feature 
tune_train <- train_bind[tri,] %>% dplyr::select(-txt,-pred1,-pred2,-pred3) %>% select(-c(1,2,3,4,5,6,7,8)) 
tune_train$deal_probability <- y
# stratafied sampling

interval <- seq(0,1,length.out = 11)
tune_train_sample = list()
for (i in 1:10) {
  sample_range <- which(tune_train$deal_probability<interval[i+1] & tune_train$deal_probability>=interval[i])
  print(length(sample_range))
  num <- round(length(sample_range)*0.01)
  tune_train_sample[[i]] <- tune_train[sample(sample_range,num),] 
}
tune_train_sample = do.call(rbind,tune_train_sample)
tune_train_sample$user_type <- as.numeric(tune_train_sample$user_type)
tune_train_sample$clust_image <- as.numeric(tune_train_sample$clust_image)
rm(tr_te) ; gc() # clean up

# setup mlr ----------------------------
cat("\n Setup Learners...  \n")
task <- mlr::makeRegrTask(id = "avito", data = tune_train_sample , target = "deal_probability")
task
rdesc <- makeResampleDesc("CV", iters = 5)


# Hyper parameter tuning --------------------------------------------------

cat("Hyper parameter tuning for xgboost...")

cat("Tune max_depth and min_child_weight...")
discrete_ps = makeParamSet(
  makeDiscreteParam("nrounds", values = c(100)),
  makeDiscreteParam("max_depth", values = c(19,21,23,25)),
  makeDiscreteParam("min_child_weight", values = c(7,9,11,13,15))
)
ctrl = makeTuneControlGrid()
res = tuneParams("regr.xgboost", task = task, resampling = rdesc,
                 par.set = discrete_ps, control = ctrl, measures = rmse)
best_max_d <- res$x$max_depth
best_min_c <- res$x$min_child_weight

cat("Tune sub_sample and colsample_bytree...")
discrete_ps = makeParamSet(
  makeDiscreteParam("nrounds", values = c(200)),
  makeDiscreteParam("subsample", values = c(0.2,0.4,0.6,0.8)),
  makeDiscreteParam("colsample_bytree", values = c(0.5,0.6,0.7,0.8)),
  makeDiscreteParam("max_depth", values = best_max_d),
  makeDiscreteParam("min_child_weight", values = best_min_c)
)
ctrl = makeTuneControlGrid()
res = tuneParams("regr.xgboost", task = task, resampling = rdesc,
                 par.set = discrete_ps, control = ctrl, measures = rmse)
best_subsam <- res$x$subsample
best_col_tree <- res$x$colsample_bytree

cat("Tune alpha and gamma...")
discrete_ps = makeParamSet(
  makeDiscreteParam("nrounds", values = c(200)),
  makeDiscreteParam("alpha", values = c(0,2)),
  makeDiscreteParam("gamma", values = c(0,0.5,1,1.5)),
  makeDiscreteParam("max_depth", values = best_max_d),
  makeDiscreteParam("min_child_weight", values = best_min_c),
  makeDiscreteParam("subsample", values = best_subsam),
  makeDiscreteParam("colsample_bytree", values = best_col_tree)
)
ctrl = makeTuneControlGrid()
res = tuneParams("regr.xgboost", task = task, resampling = rdesc,
                 par.set = discrete_ps, control = ctrl, measures = rmse)
best_alpha <- res$x$alpha
best_gamma <- res$x$gamma

cat(best_max_d,"/",best_min_c,"/",best_alpha,"/",best_subsam,"/",best_col_tree,"/",best_gamma)
#21 / 13 / 2 / 0.8 / 0.6 / 1
#23 / 15 / 2 / 0.8 / 0.6 / 1.5

#Combine data
cat("Preparing data...\n")
#2011862 x 94
X <- train_bind %>% 
  select(-txt) %>%
  select(-c(1:8)) %>%
  sparse.model.matrix(~ . - 1, .) %>%
  cbind(tfidf)
X <- X[,-grep("NONE",colnames(X))]

#-----memory manage-----
rm(tr_te, tfidf); gc()

dtest <- xgb.DMatrix(data = X[-tri, ])
X <- X[tri, ]; gc()
tri <- caret::createDataPartition(y, p = 0.9, list = F) %>% c()
dtrain <- xgb.DMatrix(data = X[tri, ], label = y[tri])
dval <- xgb.DMatrix(data = X[-tri, ], label = y[-tri])
cols <- colnames(X)

#train_ans = y[tri]
#rm(X, y, tri); gc()
#rm(X,y);gc()

cat("Training model...\n")
# p <- list(objective = "reg:logistic",
#           booster = "gbtree",
#           eval_metric = "rmse",
#           nthread = 8,
#           eta = 0.05,
#           max_depth = best_max_d ,  #21
#           min_child_weight = best_min_c, #9
#           gamma = 0,
#           subsample = best_subsam, #0.8
#           colsample_bytree = best_col_tree, #0.7
#           alpha = best_alpha, #2
#           lambda = 0,
#           nrounds = 2000)
p <- list(objective = "reg:logistic",
          booster = "gbtree",
          eval_metric = "rmse",
          nthread = 8,
          eta = 0.05,
          max_depth = 23,  
          min_child_weight = 13, 
          gamma = 1.5,
          subsample = 0.8,
          colsample_bytree = 0.6,
          alpha = 2, 
          lambda = 0,
          nrounds = 2000)
ptm <- proc.time()
#xgbcv <- xgb.cv( params = p, data = dtrain, nrounds = 2000, nfold = 5, showsd = T, stratified = T, print_every_n = 10, early_stop_round = 20, maximize = F)
#n_iter <- xgbcv$niter
m_xgb <- xgb.train (params = p, data = dtrain, nrounds = 2000, watchlist = list(val=dval,train=dtrain), print_every_n = 10, early_stop_round = 10, maximize = F)
#m_xgb <- xgb.train(p, dtrain, p$nrounds, list(val = dval), print_every_n = 10, early_stopping_rounds = 50)
proc.time() - ptm

#-----------importance plot------------
xgb.importance(cols, model = m_xgb) %>%
  xgb.plot.importance(top_n = 10)

#-----------Testing data predict------------
cat("Creating submission file...\n")
read_csv("input/sample_submission.csv")  %>%  
  mutate(deal_probability = predict(m_xgb, dtest)) %>%
  write_csv(paste0("xgb_tfidf", m_xgb$best_score, ".csv"))
