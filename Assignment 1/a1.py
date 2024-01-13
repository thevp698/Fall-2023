import pandas as pd
import re
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression

df_facts = open("facts.txt", "r")
df_fake = open("fake.txt", "r")
fact_lines = df_facts.readlines()
fake_lines = df_fake.readlines()

df_facts = pd.DataFrame(fact_lines, columns=["text"])
df_fake = pd.DataFrame(fake_lines, columns=["text"])

#add last column for labels, facts = 1, fake = 0
df_facts["Label"] = 1
df_fake["Label"] = 0

df_animal = pd.concat([df_facts, df_fake], axis=0)
train_df, test_df = train_test_split(df_animal, test_size=0.2, random_state=42)

#use stemming
def stemmization(df):
    st = PorterStemmer()
    df['text'] = df['text'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
    return df
#use lemmatization
def lemmatization(df):
    # return [lem.lemmatize(word) for word in text]
    lem = WordNetLemmatizer()
    df['text'] = df['text'].apply(lambda x: " ".join([lem.lemmatize(word) for word in x.split()]))
    return df
# train_df['text'].head()
#use naive bayes classifier from sklearn
def vectorizer(df, stopwords=True, lowercase=True, ngram_range=(1, 1), max_features=500):
    if stopwords:
        stop_words = "english"
    else:
        stop_words = None
    count_vect = CountVectorizer(lowercase=True, stop_words=stop_words, ngram_range=ngram_range, max_features=max_features)
    X = count_vect.fit_transform(df['text'])
    return X, count_vect

#Do lemmitization
train_df_lemma = lemmatization(train_df)

#Do stemming
train_df_stem = stemmization(train_df)

#vectorize lemma data
X_train_lemma, count_vect_lemma = vectorizer(train_df_lemma, stopwords=True, lowercase=True, ngram_range=(1, 1), max_features=None)

#vectorize stem data
X_train_stem, count_vect_stem = vectorizer(train_df_stem, stopwords=True, lowercase=True, ngram_range=(1, 1), max_features=None)

#vectorize bigram data
X_train_bigram, count_vect_bigram = vectorizer(train_df, stopwords=True, lowercase=True, ngram_range=(2, 2), max_features=None)

#Vectorize test data
X_test_lemma = count_vect_lemma.transform(test_df['text'])
X_test_stem = count_vect_stem.transform(test_df['text'])
X_test_bigram = count_vect_bigram.transform(test_df['text'])

#use naive bayes classifier from sklearn

#lemmatization
nb_lemma = MultinomialNB().fit(X_train_lemma, train_df['Label'])
Y_nb_lemma = nb_lemma.predict(X_test_lemma)
nb_lemma_score = accuracy_score(test_df['Label'], Y_nb_lemma)

#use stemming
nb_stem = MultinomialNB().fit(X_train_stem, train_df['Label'])
Y_nb_stem = nb_stem.predict(X_test_stem)
nb_stem_score = accuracy_score(test_df['Label'], Y_nb_stem)

#use bigram
nb_bigram = MultinomialNB().fit(X_train_bigram, train_df['Label'])
Y_nb_bigram = nb_bigram.predict(X_test_bigram)
nb_bigram_score = accuracy_score(test_df['Label'], Y_nb_bigram)

#use SVM classifier from sklearn

#lemmatization
svm_lemma = svm.SVC()
svm_lemma.fit(X_train_lemma, train_df['Label'])
Y_svm_lemma = svm_lemma.predict(X_test_lemma)
smv_lemma_score = accuracy_score(test_df['Label'], Y_svm_lemma)

#stemming
svm_stem = svm.SVC()
svm_stem.fit(X_train_stem, train_df['Label'])
Y_svm_stem = svm_stem.predict(X_test_stem)
svm_stem_score = accuracy_score(test_df['Label'], Y_svm_stem)

#bigram
svm_bigram = svm.SVC()
svm_bigram.fit(X_train_bigram, train_df['Label'])
Y_svm_bigram = svm_bigram.predict(X_test_bigram)
svm_bigram_score = accuracy_score(test_df['Label'], Y_svm_bigram)

#use logistic regression classifier from sklearn

#lemmatization
logistic_lemma = LogisticRegression()
logistic_lemma.fit(X_train_lemma, train_df['Label'])
Y_logistic_lemma = logistic_lemma.predict(X_test_lemma)
logistic_lemma_score = accuracy_score(test_df['Label'], Y_logistic_lemma)

#stemming
logistic_stem = LogisticRegression()
logistic_stem.fit(X_train_stem, train_df['Label'])
Y_logistic_stem = logistic_stem.predict(X_test_stem)
logistic_stem_score = accuracy_score(test_df['Label'], Y_logistic_stem)

#bigram
logistic_bigram = LogisticRegression()
logistic_bigram.fit(X_train_bigram, train_df['Label'])
Y_logistic_bigram = logistic_bigram.predict(X_test_bigram)
logistic_bigram_score = accuracy_score(test_df['Label'], Y_logistic_bigram)

# print("Naive Bayes Classifier")
print(f"Naive Bayes Classifier with lemmatization: {nb_lemma_score}")
print(f"Naive Bayes Classifier with stemming: {nb_stem_score}")
print(f"Naive Bayes Classifier with bigram: {nb_bigram_score}")

# print("SVM Classifier")
print(f"SVM Classifier with lemmatization: {smv_lemma_score}")
print(f"SVM Classifier with stemming: {svm_stem_score}")
print(f"SVM Classifier with bigram: {svm_bigram_score}")

# print("Logistic Regression Classifier")
print(f"Logistic Regression Classifier with lemmatization: {logistic_lemma_score}")
print(f"Logistic Regression Classifier with stemming: {logistic_stem_score}")
print(f"Logistic Regression Classifier with bigram: {logistic_bigram_score}")