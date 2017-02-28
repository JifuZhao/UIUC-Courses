####################################################################################
##########                                                 #########################
##########         STAT 542 Sping 2017 First Project       #########################
##########              IOWA Housing Prediction            #########################
##########            Jifu Zhao  &  Jinsheng Wang          #########################
##########                                                 #########################
####################################################################################

## Reference:  The data preprocessing part used some of the TA's Code.

#setwd("~/Documents/Classes_taken/STAT542/Projects_submission/")

# import required library
if (!require(dummies)) {
    install.packages("dummies")
}
if (!require(DAAG)) {
    install.packages("DAAG")
}
if (!require(randomForest)) {
    install.packages("randomForest")
}
if (!require(glmnet)) {
  install.packages("glmnet")
}
if (!require(moments)) {
  install.packages("moments")
}
# load the libraries
library(randomForest)
library(dummies)  # dummy variable
library(moments)  # skewness
library(DAAG)  # cross-validation
library(glmnet)    # ridge and lasso

# load the data set
train = read.csv('./train.csv')
test = read.csv('./test.csv')


#==========================  Here starts the data processing  ==============================
start.time = Sys.time()
# drop column of "LotFrontage", "Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature"
drop.names = c("LotFrontage", "Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature")
train = train[ , !(names(train) %in% drop.names)]
test = test[ , !(names(test) %in% drop.names)]

# find all categorical and numerical variables
data.type = sapply(train[ , -c(1, ncol(train))], class)
categorical.var = names(train)[which(c(NA, data.type, NA) == 'factor')]
numerical.var = names(train)[which(c(NA, data.type, NA) == 'integer')]

# create new feature named "NA" for categorical variables
for (i in categorical.var) {
    train[, i] = addNA(train[, i])
    test[, i] = addNA(test[, i])
}

# create new feature using the median value for numerical variables
for (i in numerical.var) {
    na.id = is.na(train[, i])
    if (!any(na.id)) {
        next
    }
    train[which(na.id), i] = median(train[, i], na.rm=TRUE)
}

# create new feature using the median value for numerical variables
for (i in numerical.var) {
    na.id = is.na(test[, i])
    if (!any(na.id)) {
        next
    }
    test[which(na.id), i] = median(train[, i], na.rm=TRUE)
}

# combine into one data frame
data = rbind(train[, -ncol(train)], test)

# transform numerical feature whose skewness is larger than 0.75
skewed.features = sapply(data[, numerical.var], skewness)
skewed.features = numerical.var[which(skewed.features > 0.75)]
for (i in skewed.features) {
    data[, i] = log(data[, i] + 1)
}

# find new categorical variable to create dummy variables
dummy.var = data.frame(dummy.data.frame(data[, categorical.var], sep='.'))
data = cbind(data, dummy.var)

# drop original categorical variables
data = data[ , !(names(data) %in% categorical.var)]

data.train = data[1:nrow(train), ]
data.test = data[(nrow(train) + 1):nrow(data), ]

data.train['SalePrice'] = train$SalePrice

# transform the response variable into log scale
data.train$SalePrice = log(data.train$SalePrice + 1)
end.time = Sys.time()
time_pre_data = end.time - start.time
cat("Data preprocessing time: ", time_pre_data)
cat('\n')  # start a new line for betteer visualizeation
#======================= This is the end of data preprocessing ============================


#=======================  linear regression with all variables  ============================
# first the cross-validation
start.time = Sys.time()
model.cv = cv.lm(data.train, SalePrice ~ ., m=5, seed=29, printit=FALSE)
#attr(model.cv, "ms")

# build model and make predictions
model = lm(SalePrice ~ ., data = data.train)
predict.test.y = predict(model, newdata=data.test)
predict.test.y = exp(predict.test.y) - 1

# make submission file
submission = read.csv('./test.csv')
submission$SalePrice = predict.test.y
write.table(submission, './mysubmission1.csv', row.names = FALSE, sep = ',')
end.time = Sys.time()
time_ls = end.time - start.time
cat("Linear regression time: ", time_ls)
cat('\n')  # start a new line for betteer visualizeation
#==========================================================================================




#======================   Lasso regression model   ========================================
# build lasso regression
start.time = Sys.time()
cv.lasso = cv.glmnet(as.matrix(data.train[, -c(1, ncol(data.train))]), 
                     data.train[, 'SalePrice'], nfolds=10)
lambda_lasso = cv.lasso$lambda.min   # this is the optimal lambda with minimal shrinkage
cat("Lambda used in lasso model: ", lambda_lasso)
cat('\n')  # start a new line for better visualizeation
#lambda_lasso, alpha = 1 is lasso regression
lasso.fit = glmnet(as.matrix(data.train[,-c(1, ncol(data.train))]), 
                   data.train[, 'SalePrice'], alpha=1, lambda=lambda_lasso)
predict.test.y = predict(lasso.fit, s=lambda_lasso, newx=as.matrix(data.test[, -1]))
predict.test.y = exp(predict.test.y) - 1

#cv.lasso$cvm
submission = read.csv('./test.csv')
submission$SalePrice = predict.test.y
write.table(submission, './mysubmission2.csv', row.names = FALSE, sep = ',')
end.time = Sys.time()
time_lasso = end.time - start.time
cat("Lasso regression time: ", time_lasso)
cat('\n')  # start a new line for betteer visualizeation
#===========================================================================================



#======================   Random Forest regression model   =================================
# build random forest regression
# build the model
start.time = Sys.time()
model = randomForest(SalePrice ~ ., data=data.train, importance=T, ntree=500)
predict.train.y = predict(model, data.train)
train.mse = sum((predict.train.y - data.train$SalePrice) ^ 2) / length(predict.train.y)

predict.test.y = predict(model, data.test)
predict.test.y = exp(predict.test.y) - 1

# make submission file
submission = read.csv('./test.csv')
submission$SalePrice = predict.test.y
write.table(submission, './mysubmission3.csv', row.names = FALSE, sep = ',')
end.time = Sys.time()
time_rf = end.time - start.time
cat("Radom forest regression time: ", time_rf)
cat('\n')  # start a new line for betteer visualizeation
#==========================================================================================


# End of this R file
#  Jifu Zhao (NPRE) & Jinsheng Wang (NPRE), 02/26/17