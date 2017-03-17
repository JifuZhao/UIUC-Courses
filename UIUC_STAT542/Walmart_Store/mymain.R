setwd('/Users/jifu/GitHub/Courses/UIUC_Courses/UIUC_STAT542/Walmart_Store/')
options(warn=-1)
# import required library
if (!require(lubridate)) {
    install.packages('lubridate')
}
if (!require(forecast)) {
    install.packages('forecast')
}
if (!require(plyr)) {
    install.packages('plyr')
}

library(lubridate)  # convert date information
library(forecast)  # make forecast
library(plyr)

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
    
    # get the unique Date information
    Date = sort(unique(train$Date))
    
    # choose the median value from the last year, in week-1, week, and week+1
    for (s in 1:n.store){
        for (d in 1:n.dept){
            
            # find the data for (store, dept) = (s, d)
            test.id = which(test$Store == store[s] & test$Dept == dept[d] &
                                test$year == year & test$month == month)
            test.temp = test[test.id, ]
            train.id = which(train$Store == store[s] & train$Dept == dept[d])
            train.temp = train[train.id, ]
            
            # skip
            if (length(test.id) == 0) {
                next
            }
            
            # cat(store[s], dept[d], '\n')
            
            # ------------------------------------------
            #             model 1
            # ------------------------------------------
            #             for (i in 1:length(test.id)){
            #                 id.1 = which(train.temp$week == test.temp[i,]$week - 1 & 
            #                              train.temp$year == test.temp[i,]$year - 1)
            #                 id.2 = which(train.temp$week == test.temp[i,]$week & 
            #                              train.temp$year == test.temp[i,]$year - 1)
            #                 id.3 = which(train.temp$week == test.temp[i,]$week + 1 & 
            #                              train.temp$year == test.temp[i,]$year - 1)
            #                 id = c(id.1, id.2, id.3)
            
            #                 # three weeks in the last year
            #                 tempSales = train.temp[id, 'Weekly_Sales']
            
            #                 if (length(tempSales) == 0){
            #                     test$Weekly_Pred1[test.id[i]] = 0
            #                 }else{
            #                     test$Weekly_Pred1[test.id[i]] = median(tempSales)
            #                 }
            #             }
            
            # ------------------------------------------
            #             model 2
            # ------------------------------------------
            if (length(train.id) < 50){
                test$Weekly_Pred2[test.id] = 0
            } else{
                tmp.left = as.data.frame(Date)
                tmp.right = train.temp[, c('Date', 'Weekly_Sales')]
                joined = join(tmp.left, tmp.right, by='Date')
                joined[is.na(joined)] = 0
                tmp.ts = ts(joined$Weekly_Sales, frequency=52)
                
                # fit a auto arima model
                fit = auto.arima(tmp.ts, seasonal=TRUE, allowdrift=TRUE)
                fc = forecast(fit, h=length(test.id))
                test$Weekly_Pred2[test.id] = fc$mean
            }
        }
    }
    
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
cat('Total used time is:\t', end.time - start.time, '\n')
