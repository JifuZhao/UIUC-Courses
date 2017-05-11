# ——————————————————————————————————
Instructions to run the code

Unzip the .zip file into your computer. Put train.csv and test.csv into the folder (don’t change the file name). Then you are ready to run the code.

Note: the code requires python3, the used packages include: warnings, numpy, pandas, surprise.

# ——————————————————————————————————
Main code: mymain.py

$: python mymain.py

The code will automatically read train and test data and train the model. Since there are 2 simple models, the running time is expected to be around 3 minutes. 

The result is saved into the file named: mysubmission1.csv and mysubmission2.csv

Note:
test.csv should have three columns: ID, user, movie

train.dat should be “::” separated file with four columns: UserID, MovieID, Rating, Timestamp



