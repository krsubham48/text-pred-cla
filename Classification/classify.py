import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.utils import shuffle

data = pd.read_csv('/Users/kr_subham/Desktop/TextClassification/reviews_imdb.tsv', delimiter='\t')
data.drop('id', axis=1, inplace=True)
data = shuffle(data)

no_html = []
def remove_html(raw_review):
	flag = BeautifulSoup(raw_review, 'html5lib')
	return flag.get_text()
for i in data['review']:
	no_html.append(remove_html(i))

text_only = []
def keep_text(raw_review):
	flag = re.sub('[^a-zA-Z]', ' ', raw_review)
	return flag
for i in no_html:
	text_only.append(keep_text(i))

lower_text = []
def lowercase(raw_review):
	return raw_review.lower()
for i in text_only:
	lower_text.append(lowercase(i))

tokenize_words = []
def tokens(raw_review):
	return raw_review.split()
for i in lower_text:
	tokenize_words.append(tokens(i))

no_stopwords = []
stopwords = set(stopwords.words('english'))
def remove_stopwords(raw_review):
	raw_review = [w for w in raw_review if not w in stopwords ]
	return raw_review
for i in tokenize_words:
	no_stopwords.append(remove_stopwords(i))

sentences = []
for i in no_stopwords:
	sentences.append(' '.join(word for word in i))
    
dict_review = {'review':[i for i in sentences], 'sentiment':[j for j in data['sentiment']]}
data_clean = pd.DataFrame(dict_review, columns=['review', 'sentiment'])
train, test = train_test_split(data_clean, test_size=0.2)
countVect = CountVectorizer(max_features=8000)
x_train_counts = countVect.fit_transform(train['review'])
x_test_counts = countVect.fit_transform(test['review'])

clf_svm = svm.SVC(kernel='rbf', C=100.0)
clf_svm.fit(x_train_counts, train['sentiment'])
test_accuracy = clf_svm.score(x_test_counts, test['sentiment'])
print('Test accuracy using SVM: ', test_accuracy)