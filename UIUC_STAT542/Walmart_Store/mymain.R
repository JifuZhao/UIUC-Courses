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

# transform into numerical value
train$IsHoliday = as.numeric(train$IsHoliday)
test$IsHoliday = as.numeric(test$IsHoliday)

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
start_date = train.week[1]
train.week = train.week - start_date  # date is now 0, 7, 14, ...
train.week = train.week / 7 + 5  # make 2010-02-05 as '5'
train.week = as.numeric(train.week) %% 52  ## 52 weeks in a year
train$week = train.week

test.week = test$Date
test.week = test.week - start_date
test.week = test.week / 7 + 5  # make 2010-02-05 as '5'
test.week = as.numeric(test.week) %% 52
test$week = test.week

start.time = Sys.time()  # get time information

for (t in 1:20) {
  month = 2 + t
  year = 2011
  if (month > 12) {
    month = month - 12
    year = 2011 + 1
  }
  
  # get the tmp test data
  tmp.test = test[(test$year == year) & (test$month == month), ]
  
  # get the length of unique store and department
  store = sort(unique(tmp.test$Store))
  n.store = length(store)
  dept = sort(unique(tmp.test$Dept))
  n.dept = length(dept)
  
  # ------------------------------------------
  #             model 1
  # ------------------------------------------
  
  # choose the median value from the last year, in week-1, week, and week+1
  for (s in 1:n.store){
    for (d in 1:n.dept){
      # find the data for (store, dept) = (s, d)
      test.id = which(test$Store == store[s] & test$Dept == dept[d] &
                        test$year == year & test$month == month)
      test.temp = test[test.id, ]
      train.id = which(train$Store == store[s] & train$Dept == dept[d])
      train.temp = train[train.id, ]
      
      # ------------------------------------------
      #             model 1
      # ------------------------------------------
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
  
  # # ------------------------------------------
  # #             model 2
  # # ------------------------------------------
  # # build model 2
  # for (d in 1:n.dept){
  #   # find the data for dept = d
  #   test.id = which(test$Dept == dept[d] & test$year == year & test$month == month)
  #   test.temp = test[test.id, ]
  #   train.id = which(train$Dept == dept[d])
  #   train.temp = train[train.id, ]
  # 
  #   # fit a linear model
  #   model = lm(Weekly_Sales ^ 0.2 ~ week + IsHoliday + month + Store, data=train.temp)
  #   # make prediction
  #   predicted = predict(model, test.temp)
  #   test$Weekly_Pred2[test.id] = predicted ^ 5
  # }
  
  # read new input file
  tmp.filename = paste('xxx', t, '.csv', sep='');
  newtest = read.csv(tmp.filename)
  
  # transform the date
  newtest$Date = as.Date(newtest$Date, '%Y-%m-%d')
  
  # get the year and month information
  newtest$year = year(newtest$Date)
  newtest$month = month(newtest$Date)
  
  # process the date
  tmp.week = newtest$Date
  tmp.week = tmp.week - start_date
  tmp.week = tmp.week / 7 + 5  # make 2010-02-05 as '5'
  tmp.week = as.numeric(tmp.week) %% 52
  newtest$week = tmp.week
  
  # merge together
  train = rbind(train, newtest[, names(train)])
}

end.time = Sys.time()
cat('Total used time is:\t', end.time - start.time, 'min\n')

# define weight w
weight = 4 * test$IsHoliday + 1

# calculate the performance of different models
WMAE1 = sum(weight * abs(test$Weekly_Pred1 - test$Weekly_Sales)) / sum(weight)
WMAE2 = sum(weight * abs(test$Weekly_Pred2 - test$Weekly_Sales)) / sum(weight)
WMAE3 = sum(weight * abs(test$Weekly_Pred3 - test$Weekly_Sales)) / sum(weight)

# output the performance of different models
cat('Model performace:\n')
cat(WMAE1, '\t', WMAE2, '\t', WMAE3, '\n')