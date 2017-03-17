setwd('/Users/jifu/GitHub/Courses/UIUC_Courses/UIUC_STAT542/Walmart_Store/')

options(warn=-1)

# import required library
if (!require(lubridate)) {
  install.packages('lubridate')
}
if (!require(forecast)) {
  install.packages('forecast')
}

library(lubridate)  # convert date information
library(forecast)  # make forecast

# load the data
train = read.csv('./train.csv')
test = read.csv('./test.csv')

# transform the date
train$Date = as.Date(train$Date, '%Y-%m-%d')
test$Date = as.Date(test$Date, '%Y-%m-%d')

# get the year and month information
train$year = year(train$Date)
test$year = year(test$Date)

train$month = month(train$Date)
test$month = month(test$Date)

# get week information
train.week = train$Date
train.week = train.week - train.week[1]  # date is now 0, 7, 14, ...
train.week = train.week / 7 + 5  # make 2010-02-05 as '5'
train.week = as.numeric(train.week) %% 52  ## 52 weeks in a year
train$week = train.week

test.week = test$Date
test.week = test.week - test.week[1]
test.week = test.week / 7 + 9 # make 2011-03-04 as '9'.
test.week = as.numeric(test.week) %% 52
test$week = test.week

start.time = Sys.time()  # get time information

# get the tmp test data
tmp.test = test

# print useful information
cat('Current is:\t', nrow(tmp.test), nrow(train), '\n')

# get the length of unique store and department
store = sort(unique(tmp.test$Store))
n.store = length(store)
dept = sort(unique(tmp.test$Dept))
n.dept = length(dept)

# choose the median value from the last year, in week - 1, week, and week + 1
for (s in 1:n.store){
  for (d in 1:n.dept){
    # find the data for (store, dept) = (s, d)
    test.id = which(test$Store == store[s] & test$Dept == dept[d])
    test.temp = test[test.id, ]
    train.id = which(train$Store == store[s] & train$Dept == dept[d])
    train.temp = train[train.id, ]
    #             cat(length(test.id), dim(test.temp), dim(train.temp), '\n')
    
    for (i in 1:length(test.id)){
      id.1 = which(train.temp$week == test.temp[i,]$week - 1 & 
                     train.temp$year == test.temp[i,]$year - 1)
      id.2 = which(train.temp$week == test.temp[i,]$week & 
                     train.temp$year == test.temp[i,]$year - 1)
      id.3 = which(train.temp$week == test.temp[i,]$week + 1 & 
                     train.temp$year == test.temp[i,]$year - 1)
      id = c(id.1, id.2, id.3)
      
      # three weeks in the last year
      tempSales = train.temp[id, 'Weekly_Sales']
      
      if (length(tempSales) == 0){
        test$Weekly_Pred1[test.id[i]] = 0
      }else{
        test$Weekly_Pred1[test.id[i]] = median(tempSales)
      }
    }
  }
}

Sys.time() - start.time