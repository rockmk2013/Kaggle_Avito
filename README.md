# Kaggle Avito Demand Prediction Competition
### This is a project on Kaggle Competition that I work with Leo Liu and Michelle Hsu.
### Competition URL : https://www.kaggle.com/c/avito-demand-prediction
## Feature Engineering
### We use the following method to extract our key features:
#### 1. Tokenized & TF-IDF method 
#### 2. SVD dimention reduction
#### 3. Image Quality Processing
#### 4. Ridge Regression Prediction
#### 5. Other Statistics Summarized Method

## Modeling

### We choose two model algorithms to train our data :
#### 1. XGBoost
#### 2. LightGBM

### LightGBM is a new model algorithm used widely on Kaggle Competition, it improved the way of growth on each decision tree(choose better node as brach) and reduce the use of memory. So it cost less time than XGBoost and has better accuracy.

### Finally, we got RMSE 0.2213 on this Competition and ranked about 30% on the leaderboard. We find both image and text feature will have importance on the model as the following figure.
![](https://github.com/rockmk2013/Kaggle_Avito/blob/master/feature_import_0.2213.png)





