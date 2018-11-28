# to save trained classifier.
import pickle
# svm classifier.
from sklearn.svm import SVC
from sklearn import model_selection
# divided data from Divide_data
from Divide_data import X_train, X_test, Y_train, Y_test,seed

# using svm classifier with linear kernel
predictiveModel = SVC(kernel='linear')
# training the model
print('Training Started . . .')
predictiveModel.fit(X_train, Y_train)
print('Training is completed')
# save the predictiveModel to disk
filename = 'predictiveModel.sav'
pickle.dump(predictiveModel, open(filename, 'wb'))




