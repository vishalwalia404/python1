import pickle
import pandas
import numpy as np
# importing divided data
from Divide_data import X_train, X_test, Y_train, Y_test

# function to predict user should visit the place or not.
# take input a numpy array of which contains features like comments,
# likes, hearts, hate, normal, caption_positivity, already_visited.
# return  0 or 1
def predictor(data):
    filename = 'predictiveModel.sav'
    # load the predictiveModel from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(X_test, Y_test)
    test = np.array([data])
    #test= test.reshape(-1, 1)
    y_pred  = loaded_model.predict(test)
    return y_pred
# printing results.
fbdata = pandas.read_csv("testdata.csv", encoding = "ISO-8859-1")
print('--------------------------------------------------------------------------')
print("%-30s %-30s %s" %('Visited By','Name of Place','Visit/Or Not'))
print('--------------------------------------------------------------------------')
for index, row in fbdata.iterrows():
    data = [row['comments'],row['likes'],
            row['heart'],row['hate'],row['normal'],row['caption_positivity'],
            row['already_visited']]
    
    print(data)
    flag =None
    if (predictor(data))[0] ==1:
        flag = 'Yes'
    else:
        flag = 'No'
    print("%-30s %-30s %s" %(row['Name'],row['Place'],flag))
    print('--------------------------------------------------------------------------')

