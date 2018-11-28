Introduction
We have trained a classifier to suggest the user that he/she should visit the place or not that is
visited by his/her Facebook friend based on the number of comments, likes, heart, hate, normal
reaction positivity in post caption. One more important feature we have considered here is if the
user has already visited the place or not. Unfortunately, there is no API currently available,
which can get the listed information from the user account. So have created dummy data, which
contains 10000 records. Date set contains the following columns.
Name: Column contains the name of the friend who has visited a place and made a check-in
activity on Facebook.
Place: Name of the place visited by the friends or name of the place mention in the check-in
activity.
Comments: Number of comment in on the post.
Likes: Number of likes on the post.
Heart: Number heart emoji on the post.
Hate: Number of angry emoji on the post.
Normal: number of sad emoji on the post.
Caption_positivity: Contains percentage positivity of post caption.
Already_visited: Indicated if the user has already visited the place or not.
Should_visit: Indicated if the user should visit the place or not which is our target.
Features Extraction
Favorable (good) Features
1. A number of likes.
2. A number of comments.
3. A number of heart emoji.
Non-favorable (Bad) Features
1. Number of angry emoji

3 | Page
2. Already visited
Normal Features
1. Number of sad emoji
2. Caption positivity
Target Vector
1. Should visit.
Data Generation and data processing
We have used an online tool to generate random data for the classification. You can found the
tool here: https://www.mockaroo.com/. Description of the columns data type is as followed.
Name: Strings
Place: Strings
Comments: Random integer values between 0 to 1000.
Likes: Random integer values between 0 to 1000.
Heart: Random integer values between 0 to 1000.
Hate: Random integer values between 0 to 1000.
Normal: Random integer values between 0 to 1000.
Caption_positivity: Random float values between 0.0 to 100.0.
Already_visited: Random value 0 or 1. 1 represents user has already visited the place and 0
represent user have not visited the place.
Target Vector Generation
To generate the target vector form the data behave generated randomly we used a python script.
Python code is written based on the following assumptions.
1. The user is never intended to visit a place twice.
2. If the average of good features is greater than average of bad features and positivity in the
caption is greater than 60 percent then user always intends to visit the place.

4 | Page
3. If the average of good features is less than average of bad features and positivity in the
caption is greater than 80 percent then the choice of the user is random mean it’s up to
him/her that he/she wants to visit the place or not.
4. If the average of good features is equal than average of bad features then the choice of the
user is random mean it’s up to him/her that he/she wants to visit the place or not.
Should_visit (Target vector): Contains 1 or 0 digit which indicates should visitor should not
visit respectively.
Implementation
After data generation, we have trained the SVM classifier on the data. SVM classifier is a part of
the sklearn library of python. In addition, we have used pandas and numpy library of python for
data manipulation.
1. Datacreation.py: This script inserts the values in the should_visit based on the
assumptions we have made.
2. Divide_data.py: This script separates features vectors and target vectors and passes to
the model_selection method which splits the data for training and testing. 80% of the
generated data is used for training and 20 percent data is used for testing.
3. predictive_model.py: Python code in this file used the training data to train SVM
classifier. Training takes 10 to 15 minutes. After the training is completed, it saves the
trained classifier in the local disk so that it can be used by loading it.
4. test.py: The code in this script load the saved classifier and test the accuracy of the
classifier on the test data we have to create Divide_data.py.
5. Extended_test.py: This code test the classifier on different data. We have created
another csv file which contains 100 records to check the robustness of the model.
