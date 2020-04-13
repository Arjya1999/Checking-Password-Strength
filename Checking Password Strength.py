import seaborn as sns
import pandas as pd

dataset=pd.read_csv('data.csv',',',error_bad_lines=False)
dataset.describe()
dataset[dataset['password'].isnull()]
dataset=dataset.dropna()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
'''
sns.set_style('whitegrid')
sns.countplot(x='strength',data=dataset,palette='RdBu_r')
'''
dataset['strength'].value_counts()
def word_divide_char(inputs):
    characters=[]
    for i in inputs:
        characters.append(i)
    return characters

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(tokenizer=word_divide_char)
X=vectorizer.fit_transform(dataset['password']).toarray()
X.shape
vocabulary=vectorizer.vocabulary_
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
'''
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty="l2",multi_class='multinomial', solver='newton-cg',random_state = 0)
classifier.fit(X_train, y_train)
'''
# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, y_pred)
'''
X_predict=np.array(["z8@@2l-f"])
X_predict=vectorizer.transform(X_predict)
Y_pred=classifier.predict(X_predict)
print(Y_pred)
'''
import pickle
# open a file, where you ant to store the data
file = open('checking_password_strength.pkl', 'wb')
# dump information to that file
pickle.dump(classifier, file)
pickle.dump(vectorizer,open('td-idf.pkl', 'wb'))