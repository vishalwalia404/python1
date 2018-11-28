import pickle
import pandas
import numpy as np
# to display the report of the model.
from sklearn.metrics import classification_report
# importing divided data.
from Divide_data import X_train, X_test, Y_train, Y_test

# class labels
target_names = ['Should Not Visit', 'Should Visit']
filename = 'predictiveModel.sav'
# load the predictiveModel from disk
loaded_model = pickle.load(open(filename, 'rb'))
# accuracy of the model.
result = loaded_model.score(X_test, Y_test)
# prediction
y_pred  = loaded_model.predict(X_test)
print('True values:')
print(Y_test)
print('Predicted values:')
print(y_pred)
# classification report.
print(classification_report(Y_test, y_pred))
print('Accuracy: ',result)
