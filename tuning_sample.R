#------XGB tuning-----

# Tune XGB model
# number of Classes in Species
m = nlevels(test_model2$isagain)
#1. Tune nrounds

# Set parameters
param <- list(booster = "gbtree", objective = "binary:logistic", eval_metric = "error",  eta = 0.1, gamma = 0, max_depth = 3, min_child_weight = 5, subsample=0.8, colsample_bytree=0.7)
# Perform cross validation
xgb_meta_cv <- xgb.cv( params = param, data = trainMatrix, nrounds = 1000, nfold = 5, label = train_ans, showsd = T, stratified = T, early_stopping_rounds = 50, maximize = F, print_every_n = 5)
# Keep the best number of iterations
best_iter <- xgb_meta_cv$best_iteration

#2.1 Tune max_depth and min_child_weight
# XGBoost
# Set control
xgb_trcontrol_1 = trainControl(method = "cv", number = 3, allowParallel = TRUE)

# xgFit1 = train(Species ~ ., data = iris.Train,
# 		trControl = ctrl, method = "xgbTree", num_class = m )

# Set grid and train
xgb_grid_1 <- expand.grid(nrounds=best_iter, eta = 0.1, gamma = 0, max_depth = c(3,5,7,9), min_child_weight = c(1,3,5,7,9), subsample=0.8, colsample_bytree=0.7)

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