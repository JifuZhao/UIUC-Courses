
all.start.time = Sys.time()

options(warn=-1)

# import required library
if (!require(dummies)) {
    install.packages("dummies")
}
if (!require(DAAG)) {
    install.packages("DAAG")
}
if (!require(xgboost)) {
    install.packages("xgboost")
}
if (!require(randomForest)) {
    install.packages("randomForest")
}
if (!require(gbm)) {
    install.packages("gbm")
}
if (!require(glmnet)) {
    install.packages("glmnet")
}
library(xgboost)       # XGBoost model
library(randomForest)  # Random Forest model
library(gbm)           # GBM model
library(dummies)       # Dummy variable
library(DAAG)          # Cross-Validation
library(glmnet)        # Logistic regression with Lasso

# -----------------------------------------------------------
# load the data set
train = read.csv('./train1.csv')
test = read.csv('./test1.csv')

# the ID as the first column
TEST_predict = data.frame(test[, "id"])  
# the SalePrice was set to 0.0 initially
TEST_predict['prob'] = matrix(rep(0.0, nrow(test))) 
# reset the column names of final results
colnames(TEST_predict) = c("id", "prob")  

# -----------------------------------------------------------
# Pre-processing and feature selection

# change the loan status
default.y = c('Default', 'Charged Off', 'Late (31-120 days)', 'Late (16-30 days)', 
              'Does not meet the credit policy. Status:Charged Off')
train$loan_status = ifelse(train$loan_status %in% default.y, 1, 0)

# -----------------------------------------------------------
# drop columns that have too many missing values
drop.names = c("mths_since_last_delinq", "mths_since_last_record", "mths_since_last_major_derog", 
               "annual_inc_joint", "dti_joint", "open_acc_6m", "open_il_6m", "open_il_12m", 
               "open_il_24m", "mths_since_rcnt_il", "total_bal_il", "il_util", "open_rv_12m", 
               "open_rv_24m", "max_bal_bc", "all_util", "inq_fi", "total_cu_tl", "inq_last_12m")
train = train[, !(names(train) %in% drop.names)]
test = test[, !(names(test) %in% drop.names)]

# Transform categorical feature into numerical feature
train$sub_grade = as.numeric(train$sub_grade)
test$sub_grade = as.numeric(test$sub_grade)

# -----------------------------------------------------------
# Drop useless features
# drop useless categorical features
drop.names = c("emp_title", "issue_d", "pymnt_plan", "url", "desc", "title", "zip_code", "addr_state", 
               "earliest_cr_line", "last_pymnt_d", "next_pymnt_d", "last_credit_pull_d", "application_type", 
               "verification_status_joint", "grade")
train = train[, !(names(train) %in% drop.names)]
test = test[, !(names(test) %in% drop.names)]

# drop useless numerical features
drop.names = c("id", "member_id", "policy_code")
train = train[, !(names(train) %in% drop.names)]
test = test[, !(names(test) %in% drop.names)]

# -----------------------------------------------------------
# Data Structures
# find all categorical and numerical variables
data.type = sapply(train, class)
cat.var = names(train)[which(data.type == 'factor')]
num.var = names(train)[which((data.type == 'integer') | (data.type == 'numeric'))]

# -----------------------------------------------------------
# Fill NA values
# create new feature named "NA" for categorical variables
for (i in cat.var) {
    train[, i] = addNA(train[, i])
    test[, i] = addNA(test[, i])
}

# create new feature using the median value for numerical variables
for (i in num.var) {
    na.id = is.na(train[, i])
    tmp.median = median(train[, i], na.rm=TRUE)
    train[which(na.id), i] = tmp.median
    
    na.id = is.na(test[, i])
    test[which(na.id), i] = tmp.median
}

# -----------------------------------------------------------
# Skewness
skew = c('installment', 'annual_inc', 'dti', 'revol_bal', 'total_acc', 'out_prncp', 'out_prncp_inv', 
         'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 
         'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt', 'tot_coll_amt', 'tot_cur_bal',
         'total_rev_hi_lim')

for (i in skew){
    train[, i] = log(train[, i] + 1)
    test[, i] = log(test[, i] + 1)
}

# -----------------------------------------------------------
# Dummy Variables
train.y = train$loan_status
data = rbind(train[, names(test)], test)

# create dummy variables
dummy.var = data.frame(dummy.data.frame(data[, cat.var], sep='.'))
data = cbind(data, dummy.var)

# drop original categorical variables
data = data[ , !(names(data) %in% cat.var)]

# split into train and test data set
train = data[1:nrow(train), ]
test = data[(nrow(train) + 1): nrow(data), ]
train['loan_status'] = train.y

scale = function(score) {
    # function to scale the score
    score[which(score < 1e-15)] = 1e-15
    score[which(score > 1 - 1e-15)] = 1 - 1e-15
    score
}

# temporarily code
test$loan_status = ifelse(test$loan_status %in% default.y, 1, 0)

start.time = Sys.time()  # get time information

# fit the logistic model
log.model = glm(loan_status ~ ., data=train, family=binomial)

end.time = Sys.time()
cat("Logistic Regression Model\n")
cat('Start time:\t', as.character(start.time), '\n')
cat('End time:\t', as.character(end.time), '\n')
cat('Total time:\t', end.time - start.time, '\n')

# calculate log-loss
test.y.hat = predict(log.model, test, type="response")
test.y.hat = scale(test.y.hat)

# # make submission file
# TEST_predict$prob = test.y.hat
# write.table(TEST_predict, './mysubmission1.txt', row.names=FALSE, sep=',')

# temporarily code
train.y.hat = predict(log.model, train, type="response")
train.y.hat = scale(train.y.hat)

train.y = train$loan_status
test.y = test$loan_status

train.loss = -sum(train.y * log(train.y.hat)) / length(train.y)
test.loss = -sum(test.y * log(test.y.hat)) / length(test.y)

cat('train loss\t', train.loss, '\n')
cat('test loss\t', test.loss, '\n')

start.time = Sys.time()  # get time information

# build the adaboost model
ada.boost = gbm(loan_status ~ ., data=train, distribution="adaboost", n.trees=1000, 
                shrinkage=0.5, bag.fraction=1, cv.folds=5, n.cores=8)
gbm.perf(ada.boost, method='cv', plot.it=T)  # estimating the best iterations

end.time = Sys.time()
cat("GBM Model\n")
cat('Start time:\t', as.character(start.time), '\n')
cat('End time:\t', as.character(end.time), '\n')
cat('Total time:\t', end.time - start.time, '\n')

# calculate log-loss
test.y.hat = predict(ada.boost, test, type="response")
test.y.hat = scale(test.y.hat)

# # make submission file
# TEST_predict$prob = test.y.hat
# write.table(TEST_predict, './mysubmission1.txt', row.names=FALSE, sep=',')

# temporarily code
train.y.hat = predict(ada.boost, train, type="response")
train.y.hat = scale(train.y.hat)

train.y = train$loan_status
test.y = test$loan_status

train.loss = -sum(train.y * log(train.y.hat)) / length(train.y)
test.loss = -sum(test.y * log(test.y.hat)) / length(test.y)

cat('train loss\t', train.loss, '\n')
cat('test loss\t', test.loss, '\n')

start.time = Sys.time()  # get time information

# build the random forest model
rf.model = randomForest(as.factor(loan_status) ~ ., data=train, importance=T, ntree=100)

end.time = Sys.time()
cat("Random Forest Model\n")
cat('Start time:\t', as.character(start.time), '\n')
cat('End time:\t', as.character(end.time), '\n')
cat('Total time:\t', end.time - start.time, '\n')

# plot the error rate
tmp = rf.model$err.rate
plot(c(0, rf.model$ntree), range(tmp), type="n",
     xlab = "Number of trees", ylab="Error")
lines(tmp[,1], col="black")
lines(tmp[,2], col="red")
lines(tmp[,3], col="green")

# calculate log-loss
test.y.hat = predict(rf.model, test, type="prob")[, 2]
test.y.hat = scale(test.y.hat)

# # make submission file
# TEST_predict$prob = test.y.hat
# write.table(TEST_predict, './mysubmission1.txt', row.names=FALSE, sep=',')

# temporarily code
train.y.hat = predict(rf.model, train, type="prob")[, 2]
train.y.hat = scale(train.y.hat)

train.y = train$loan_status
test.y = test$loan_status

train.loss = -sum(train.y * log(train.y.hat)) / length(train.y)
test.loss = -sum(test.y * log(test.y.hat)) / length(test.y)

cat('train loss\t', train.loss, '\n')
cat('test loss\t', test.loss, '\n')

train.m.y = train$loan_status
train.m.x = train[, -which(names(train) == "loan_status")]
train.m.x = apply(train.m.x, 2, as.numeric)

test.m.x = test[, -which(names(test) == "loan_status")]
test.m.x = apply(test.m.x, 2, as.numeric)

start.time = Sys.time()  # get time information

# build XGBoost model
set.seed(2017)
cv.dummy = xgb.cv(data=train.m.x, label=train.m.y, nfold=5, max.depth=6, eta=0.5, 
                  nrounds=200, nthread=8, early.stop.round=3, verbose=T,
                  objective="binary:logistic", metrics=list("logloss"))

end.time = Sys.time()
cat("XGBoost Model\n")
cat('Start time:\t', as.character(start.time), '\n')
cat('End time:\t', as.character(end.time), '\n')
cat('Total time:\t', end.time - start.time, '\n')

start.time = Sys.time()  # get time information

bst = xgboost(data=train.m.x, label=train.m.y, max.depth=6, eta=0.5, nrounds=85, 
              nthread=8, objective="binary:logistic", verbose=F)

end.time = Sys.time()
cat("XGBoost Model\n")
cat('Start time:\t', as.character(start.time), '\n')
cat('End time:\t', as.character(end.time), '\n')
cat('Total time:\t', end.time - start.time, '\n')

# calculate log-loss
test.y.hat = predict(bst, test.m.x)
test.y.hat = scale(test.y.hat)

# # make submission file
# TEST_predict$prob = test.y.hat
# write.table(TEST_predict, './mysubmission1.txt', row.names=FALSE, sep=',')

# temporarily code
train.y.hat = predict(bst, train.m.x)
train.y.hat = scale(train.y.hat)

train.y = train$loan_status
test.y = test$loan_status

train.loss = -sum(train.y * log(train.y.hat)) / length(train.y)
test.loss = -sum(test.y * log(test.y.hat)) / length(test.y)

cat('train loss\t', train.loss, '\n')
cat('test loss\t', test.loss, '\n')

all.end.time = Sys.time()
cat("All running time\n")
cat('Start time:\t', as.character(all.start.time), '\n')
cat('End time:\t', as.character(all.end.time), '\n')
cat('Total time:\t', all.end.time - all.start.time, '\n')


