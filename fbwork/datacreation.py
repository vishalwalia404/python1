'''
This python script is to add data in shoud vistied column based on the assumptions
'''
# importing libraries.
import csv
import pandas as pd
import random
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names
import pickle


# method to calculate features in the captions
def word_feats(words):
    return dict([(word, True) for word in words])

# readind text file which contains positive words.
with open('positive.txt') as f:
    # reading file line by line
    positive_vocab = f.readlines()
    # removing blank space.
positive_vocab = [x.strip() for x in positive_vocab]

# readind text file which contains negative words.
with open('negative.txt') as f:
    # reading file line by line
    negative_vocab = f.readlines()
    # removing blank space.
negative_vocab = [x.strip() for x in negative_vocab]

# readind text file which contains normal words.
with open('normal.txt')as f:
    # reading file line by line
    neutral_vocab = f.readlines()
    # removing blank space.
neutral_vocab = [x.strip() for x in neutral_vocab]

# assigining labels to each word.
# pos for positive.
# neg for negative.
# neu for neutral.
positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]
neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]
# dataset for training, 
train_set = negative_features + positive_features + neutral_features
# trainig the classifier. 
classifier = NaiveBayesClassifier.train(train_set)

# method to calculate and return the positivity in a given caption.
def positivity_checker(row):
    # negativity
    neg = 0
    # positivity
    pos = 0
    # caption text.
    sentence = row['caption_text']
    # coverting to lowercase.
    sentence = sentence.lower()
    normal_quote = 40
    # spliting the sentence into words
    words = sentence.split(' ')
    # each word in sentence.
    for word in words:
        # detecting freature.
        classResult = classifier.classify( word_feats(word))
        # for negative.
        if classResult == 'neg':
            neg = neg + 1
        # for positive 
        if classResult == 'pos':
            pos = pos + 1
    # percentage of positivity in the sentence.
    return round(((float(pos)/len(words))),2)*100 + normal_quote

#reading csv file without target column.
facbook_data= pd.read_csv("facebook.csv", encoding = "ISO-8859-1")
# printing data.
print(facbook_data.shape)
print(facbook_data.head())

# changing data set to numeric dataset.
facbook_data[['comments','likes','heart','hate','normal','caption_positivity',
              'already_visited']] = facbook_data[['comments','likes','heart','hate','normal','caption_positivity',
              'already_visited']].apply(pd.to_numeric)

# method to translate feature vector into target vector.
def translator(row):
    
    GoOrNor = [0, 1]
    should_visit = 0
    # ready visited not visit again.
    if row['already_visited'] == 1:
        should_visit = 0
    else:
        # average good conditions is great than bad conditions.
        avegGood = row['comments']+((row['likes']+row['heart']))/2
        avegBad = ((row['hate']+row['normal']+row['caption_positivity']))/3
        if avegGood > avegBad:
            # caption positivity is greater than 60%
            if row['caption_positivity']>60:
                should_visit = 1
                # average good conditions is great than bad conditions.
        elif avegGood < avegBad:
            if row['caption_positivity']>80:
                should_visit = random.choice(GoOrNor)
            else:
                should_visit=0
        else:
            
            should_visit = random.choice(GoOrNor)
    return should_visit

facbook_data['caption_positivity'] = facbook_data.apply (lambda row: positivity_checker(row),axis=1)
facbook_data['should_visit'] = facbook_data.apply (lambda row: translator (row),axis=1)
print(facbook_data.shape)
print(facbook_data[1:100])
# write dataset.
facbook_data.to_csv('facebook.csv', encoding='utf-8')
                    
    
