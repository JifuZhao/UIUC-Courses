setwd('/Users/jifu/GitHub/Courses/UIUC_Courses/UIUC_STAT542/Walmart_Store/')

start.time = Sys.time()  # get time information

train=read.csv("train.csv")
test = read.csv("test.csv")

source("mymain.R")

for(t in 1:2){
    
    # predict the weekly sales for month t, 
    # e.g., month 1 --> 2011-03, and month 20 --> 2012-10. 
    predict();
    
    # newtest: sales data for this month; taking the
    # same format as "train". 
    tmp.filename = paste('xxx', t, '.csv', sep='');
    newtest = read.csv(tmp.filename) 
    
    # Evaluate your prediction accuracy for this month
    # xxxx
    # xxxx
}

# ------------------------------------------
# helper part
end.time = Sys.time()
cat('Start time:\t', as.character(start.time), '\n')
cat('End time:\t', as.character(end.time), '\n')
cat('Total time:\t', end.time - start.time, '\n\n')

# ------------------------------------------
# evaluation
# ------------------------------------------

# define weight w
weight = 4 * test$IsHoliday + 1

# calculate the performance of different models
WMAE1 = sum(weight * abs(test$Weekly_Pred1 - test$Weekly_Sales)) / sum(weight)
WMAE2 = sum(weight * abs(test$Weekly_Pred2 - test$Weekly_Sales)) / sum(weight)
WMAE3 = sum(weight * abs(test$Weekly_Pred3 - test$Weekly_Sales)) / sum(weight)

# output the performance of different models
cat('Comparison of three models\n')
cat('Model 1', '\t', 'Model 2', '\t', 'Model 3', '\n')
cat(WMAE1, '\t', WMAE2, '\t', WMAE3, '\n')