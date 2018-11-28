# dividing data for model.
from sklearn import model_selection
import pandas
import numpy as np

# reading csv file which contains data.
fbdata = pandas.read_csv("facebook.csv")
# creating value arrays.
fbArray = fbdata.values
# extracting features 
X = fbArray[:,3:10]

# target vector.
Y = fbArray[:,10]

Y=Y.astype('int')
# 20% data is for training and 30% is for testing.
test_size = 0.20
seed = 7
# randomly crate training and testing data.
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
