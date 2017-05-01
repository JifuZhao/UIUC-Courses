# ——————————————————————————————————
Instructions to run the code

Unzip the .zip file into your computer. Put labeledTrainData.tsv and testData.tsv into the folder (don’t change the file name). Then you are ready to run the code.

Note: the code requires python3, the used packages include: re, numpy, pandas, matplotlib, bs4, sklearn.


# ——————————————————————————————————
Part I: mymain.py

$: python mymain.py

The code will automatically read train and test data and train the model. Since there are 6 models, the running time is expected around 4 hours (depend on you computer). The result is saved into the file named: mysubmission.csv


# ——————————————————————————————————
Part II: visualization.py

$: python visualization.py

By default, the code will automatically read the testData.tsv and make predictions on the first 50 reviews. If you want to make predictions on other file, you need to specify the file name. The result is saved into the file named: visualization.html


