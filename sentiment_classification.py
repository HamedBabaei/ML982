#clean dataset mudole
from CleanDataset import preprocessing
#pandas library for working with dataframes 
import pandas as pd

#loading dataset
train = pd.read_csv("train_df.csv")
test = pd.read_csv("test_df.csv")

# shape of dataset
print("Size of train-set is : {}".format(train.shape[0]))
print("Size of test-set is : {}".format(test.shape[0]))

#get train and test labels
X_train, X_test = train['tweets'].tolist(), test['tweets'].tolist()
y_train, y_test = train['labels'].tolist(), test['labels'].tolist()

#cleaning train and test set
X_train = [preprocessing(x) for x in X_train ]
X_test = [preprocessing(x) for x in X_test ]

#importing a vectorizer for BOW
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
#learning a vectorizer from train using fit and using transform 
#we will transform any text into vectors using learned vectorizer
X_train_data = vectorizer.fit_transform(X_train)
X_test_data = vectorizer.transform(X_test)


#call scikit-learn library for naive bayse model
from sklearn.naive_bayes import MultinomialNB
#define an instance
classifier = MultinomialNB()
#train a model on train data
classifier.fit(X_train_data, y_train)
#make a prediction on test set
predict = classifier.predict(X_test_data)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#accuracy of the model
print("Accuracy:", accuracy_score(predict, y_test))
#evaluation report
print(classification_report(predict, y_test))

print("Confusion Matrix:",confusion_matrix(predict, y_test))

