#######################
# Project 5 Test Code #
#######################

setwd("/Users/jifu/GitHub/Courses/UIUC_Courses/UIUC_STAT542/Recommender_System")

##
# Root Mean Square Error
##
rmse = function(y, yHat){

    if (length(y) != length(yHat)){
        stop('The number of predictions is wrong.')
    }
    
    if (any(!is.numeric(yHat))){
        stop('Prediction contains non-numeric values')
    }
    
    sqrt(mean((y - yHat)^2))
    
}

##
# Running your code
##
# start.time = Sys.time()
# source('mymain.R')
# end.time = Sys.time()
# run.time = as.numeric(difftime(end.time, start.time, units = 'min'))

##
# Test the accuracy of your code
##
submitFiles = paste('mysubmission', 1:2, '.csv', sep = '')

# calculate the test error on the test set
label = read.csv('label.csv', sep = ',')  # True ratings

err = rep(NA, 2)
for (met in 1:2){

    prediction = read.csv(submitFiles[met], sep = ',')
    err[met] = rmse(label$rating, prediction$rating)
    
}

write.table(err, file = 'proj_5.csv', sep = ',', row.names = FALSE,
            col.names = FALSE)
# write.table(run.time, file = 'proj_5.csv', sep = ',', 
#             row.names = FALSE, col.names = FALSE, append = TRUE)
