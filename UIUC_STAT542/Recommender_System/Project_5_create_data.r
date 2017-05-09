###############################################
# Create training and test data for project 5 #
###############################################

# You need the ratings.dat file in the working directory. 
# After running this code, you will see three files in the working dir.
# 1. train.dat
# 2. test.csv
# 3. trueLabel.csv
# You should only use the train.dat and test.csv in mymain.R. 

setwd("/Users/jifu/GitHub/Courses/UIUC_Courses/UIUC_STAT542/Recommender_System")

# These are the parameters that you can change to test your code.
mySeed = 100
set.seed(mySeed)
user.prop = 0.001  # percent of users that will only appear in test data
movie.prop = 0.005  # percent of movies that will only appear in test data
new.prop = 0.5  # percent of those ratings in the test data. 

# Read the data. 
ratings = read.table('./data/ratings.dat', sep = ':', 
    colClasses = c('integer', 'NULL'), header = FALSE
    )
colnames(ratings) = c('UserID', 'MovieID', 'Rating', 'Timestamp')

n = nrow(ratings)

# remove some users that will be included only in test data
n.user = length(unique(ratings$UserID))
test.user.id = sample(n.user, floor(n.user * user.prop))

# remove some movies that will be included only in test data
n.movie = length(unique(ratings$MovieID))
test.movie.id = sample(n.movie, floor(n.movie * movie.prop))

# test data id
test.id = with(ratings, 
             which(UserID %in% test.user.id | MovieID %in% test.movie.id)
             )

test.id = sample(test.id, round(length(test.id)) * new.prop)
             
tmp.id = NULL
if (floor(n * 0.2) - length(test.id) > 0){
    tmp.id = sample(setdiff(1:n, test.id), 
                    floor(n * 0.2) - length(test.id)
                   )
}
test.id = sort(c(tmp.id, test.id))

# training data id
train.id = setdiff(1:n, test.id)
# after removing the test id (20%), need 3/4 (0.8 * 0.75) part of data.
train.id = sample(train.id, floor(length(train.id)) * 0.75)  
train.id = sort(train.id)

# write data in files
train = ratings[train.id, ]
write.table(train, file = 'train.dat', sep = '::', row.names = FALSE,
    col.names = FALSE
)

test = ratings[test.id, ]
test$Timestamp = NULL
test$ID = 1:nrow(test)
label = test[c('ID', 'Rating')]
test$Rating = NULL
test = test[c('ID', 'UserID', 'MovieID')]
colnames(test) = c('ID', 'user', 'movie')
colnames(label) = c('ID', 'rating')

write.table(test, file = 'test.csv', sep = ',', row.names = FALSE)
write.table(label, file = 'label.csv', sep = ',', row.names = FALSE)
