#!/usr/bin/env python
# coding: utf-8

# # e-Mail Spam Classifier

import pandas as pd
import numpy as np

mail_data = pd.read_csv('~/Desktop/Datasets/mail_data.csv')
mail_data
mail = mail_data
mail
mail.head()
mail.tail()

# # Exploratory Data Analysis (EDA)

mail.describe()
mail.info()
mail.shape
mail.size


mail['Category'].unique()
mail['Message'].unique()
mail['Category'].value_counts()
mail['Message'].value_counts()
mail.isnull().sum()


mail.iloc[0].Category
mail.iloc[4786].Category
mail.iloc[0].Message
mail.iloc[4786].Message

# ### Renaming our Columns
mail.rename(columns={'Category':'target_values','Message':'text'},inplace=True)
mail.head(2)

from sklearn.preprocessing import LabelEncoder
text_encoding = LabelEncoder()
mail['target_values'] = text_encoding.fit_transform(mail['target_values'])
mail.head()

# #### ham - 0 and spam - 1

mail.iloc[0].target_values
mail.iloc[5567].target_values
mail.isnull().sum()
mail.duplicated().sum()
mail = mail.drop_duplicates(keep='first')
mail.duplicated().sum()
mail['target_values'].value_counts()

# # Data Visualisation

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
sns.countplot(mail['target_values'])
plt.show()

mail['target_values'].value_counts()
mail['text'].value_counts()
plt.pie(mail['target_values'].value_counts(), labels=['ham - 0','spam - 1'],autopct="%0.1f")
plt.show()


import nltk
nltk.download('punkt')

#No of words
mail['total_chars'] = mail['text'].apply(len)
#No of words
mail['total_words'] = mail['text'].apply(lambda m:len(nltk.word_tokenize(m)))
#No of Sentences
mail['total_sentences'] = mail['text'].apply(lambda x:len(nltk.sent_tokenize(x)))
mail.head(3)

mail.describe()
mail[['total_chars','total_words','total_sentences']].describe()
mail[mail['target_values'] == 0][['total_chars','total_words','total_sentences']].describe()
mail[mail['target_values'] == 1][['total_chars','total_words','total_sentences']].describe()

plt.figure(figsize=(10,6))
sns.histplot(mail[mail['target_values'] == 0]['total_chars'])
sns.histplot(mail[mail['target_values'] == 1]['total_chars'],color='yellow')
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(mail[mail['target_values'] == 0]['total_words'])
sns.histplot(mail[mail['target_values'] == 1]['total_words'],color='yellow')
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(mail[mail['target_values'] == 0]['total_sentences'])
sns.histplot(mail[mail['target_values'] == 1]['total_sentences'],color='yellow')
plt.show()

plt.figure(figsize=(12,10))
sns.heatmap(mail.corr(),annot=True)
plt.show()

sns.lmplot(data=mail, x="total_chars", y="total_words", col="target_values")
plt.show()

sns.pairplot(mail,hue='target_values')
plt.show()

from nltk.corpus import stopwords

import string
#stopwords.words('english')

from nltk.stem.porter import PorterStemmer
port_stem = PorterStemmer()
port_stem.stem('laughing')
string.punctuation


def transforming(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    random = []
    for x in text:
        if x.isalnum():
            random.append(x)

    text = random[:]
    #need clone the data
    random.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            random.append(i)
    
    text = random[:]
    random.clear()
    
    for i in text:
        random.append(port_stem.stem(i))
            
    return " ".join(random)

transforming('HIIi how ARe you bRo 19?')
mail['text']
transforming('Go until jurong point, crazy.. Available only')
mail['text'].apply(transforming)
mail['transformed'] = mail['text'].apply(transforming)
mail.head(2)

from wordcloud import WordCloud
word_count = WordCloud(width=500,height=500,min_font_size=10,background_color='black')

ham_word_count = word_count.generate(mail[mail['target_values'] == 0]['transformed'].str.cat(sep=" "))
plt.figure(figsize=(15,10))
plt.imshow(ham_word_count)
plt.show()

spam_word_count = word_count.generate(mail[mail['target_values'] == 1]['transformed'].str.cat(sep=" "))
plt.figure(figsize=(15,10))
plt.imshow(spam_word_count)
plt.show()
mail.head(2)

from collections import Counter
#for Spsam
no_spam_corpus = []
for message in mail[mail['target_values'] == 1]['transformed'].tolist():
    for word_cnt in message.split():
        no_spam_corpus.append(word_cnt)  
#for Ham
no_ham_corpus = []
for message in mail[mail['target_values'] == 0]['transformed'].tolist():
    for word_cnt in message.split():
        no_ham_corpus.append(word_cnt)

len(no_spam_corpus)
len(no_ham_corpus)      

plt.figure(figsize=(18,6))
sns.barplot(pd.DataFrame(Counter(no_spam_corpus).most_common(30))[0],pd.DataFrame(Counter(no_spam_corpus).most_common(30))[1])
plt.xticks(rotation='horizontal')
plt.show()

plt.figure(figsize=(18,6))
sns.barplot(pd.DataFrame(Counter(no_ham_corpus).most_common(30))[0],pd.DataFrame(Counter(no_ham_corpus).most_common(30))[1])
plt.xticks(rotation='horizontal')
plt.show()

mail.head(2)


from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf = TfidfVectorizer(max_features=3000)
X = tf_idf.fit_transform(mail['transformed']).toarray()
X
X.shape
mail.shape
Y = mail['target_values'].values
Y

from sklearn.model_selection import train_test_split
X_mailtrain, X_mailtest, Y_mailtrain, Y_mailtest = train_test_split(X,Y,test_size=0.2,random_state=2)

X_mailtrain.shape
Y_mailtrain.shape
X_mailtest.shape
Y_mailtest.shape


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB,CategoricalNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()
cnb = CategoricalNB()

# Using GaussianNB
gnb.fit(X_mailtrain,Y_mailtrain)
pred_1 = gnb.predict(X_mailtest)
gnb_acc = (accuracy_score(Y_mailtest,pred_1))
gnb_psc = (precision_score(Y_mailtest,pred_1))
print("Accuracy Score is:",gnb_acc*100)
print("Precision Score is:",gnb_psc*100)
gnb_matrix = (confusion_matrix(Y_mailtest,pred_1))
print("Confuison Matrix is:",gnb_matrix)

# Using MultinomialNB
mnb.fit(X_mailtrain,Y_mailtrain)
pred_2 = mnb.predict(X_mailtest)
mnb_acc = (accuracy_score(Y_mailtest,pred_2))
mnb_psc = (precision_score(Y_mailtest,pred_2))
print("Accuracy Score is:",mnb_acc*100)
print("Precision Score is:",mnb_psc*100)
mnb_matrix = (confusion_matrix(Y_mailtest,pred_2))
print("Confuison Matrix is:",mnb_matrix)

# Using BernoulliNB
bnb.fit(X_mailtrain,Y_mailtrain)
pred_3 = bnb.predict(X_mailtest)
bnb_acc = (accuracy_score(Y_mailtest,pred_3))
bnb_psc = (precision_score(Y_mailtest,pred_3))
print("Accuracy Score is:",bnb_acc*100)
print("Precision Score is:",bnb_psc*100)
bnb_matrix = (confusion_matrix(Y_mailtest,pred_3))
print("Confuison Matrix is:",bnb_matrix)


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)

classifiers = {
    'Support Vector Machine' : svc,
    'K-Nearest Neighbors' : knc, 
    'Naive Bayes': mnb, 
    'Decision Tree': dtc, 
    'Logistic Regression': lrc, 
    'Random Forest': rfc, 
    'AdaBoost': abc, 
    'Bagging Classifier': bc, 
    'Extra Trees Classifier': etc,
    'Gradient Boosting Classifier':gbdt
}

def training_data(classifiers,X_mailtrain,Y_mailtrain,X_mailtest,Y_mailtest):
    classifiers.fit(X_mailtrain,Y_mailtrain)
    pred = classifiers.predict(X_mailtest)
    accuracy = accuracy_score(Y_mailtest,pred)
    precision = precision_score(Y_mailtest,pred)
    
    return accuracy,precision

training_data(svc,X_mailtrain,Y_mailtrain,X_mailtest,Y_mailtest)
training_data(lrc,X_mailtrain,Y_mailtrain,X_mailtest,Y_mailtest)
training_data(mnb,X_mailtrain,Y_mailtrain,X_mailtest,Y_mailtest)


accuracy_scores = []
precision_scores = []

for name,clf in classifiers.items():
    
    current_accuracy,current_precision = training_data(clf,X_mailtrain,Y_mailtrain,X_mailtest,Y_mailtest)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)



performance_pres = pd.DataFrame({'Algorithm':classifiers.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)
performance_pres
performance_acc = pd.DataFrame({'Algorithm':classifiers.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Accuracy',ascending=False)
performance_acc
new_data = pd.melt(performance_pres, id_vars = "Algorithm")
new_data

plt.figure(figsize=(10,10))
sns.catplot(x = 'Algorithm', y='value', hue = 'variable',data=new_data, kind='bar',height=5)
plt.ylim(0.6,1.0)
plt.xticks(rotation='vertical')
plt.show()


new_data1 = pd.DataFrame({'Algorithm':classifiers.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)
new_data1 = pd.DataFrame({'Algorithm':classifiers.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores}).sort_values('Precision_scaling',ascending=False)
new_data2 = performance_pres.merge(new_data1,on='Algorithm')
new_data3 = new_data2.merge(new_data1,on='Algorithm')
new_data1 = pd.DataFrame({'Algorithm':classifiers.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars',ascending=False)
performance_pres
performance_acc
new_data3.merge(new_data1, on='Algorithm')

svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
rfc = RandomForestClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier
voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc), ('rf', rfc)],voting='soft')
voting.fit(X_mailtrain,Y_mailtrain)
final_pred = voting.predict(X_mailtest)
print("Accuracy",accuracy_score(Y_mailtest,final_pred))
print("Precision",precision_score(Y_mailtest,final_pred))
estimation=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimation=RandomForestClassifier()
classify = StackingClassifier(estimators=estimation, final_estimator=final_estimation)
classify.fit(X_mailtrain,Y_mailtrain)
predict_final = classify.predict(X_mailtest)
print("Accuracy",accuracy_score(Y_mailtest,predict_final))
print("Precision",precision_score(Y_mailtest,predict_final))

import pickle
pickle.dump(tf_idf,open('spam.pkl','wb'))
pickle.dump(mnb,open('spammodel.pkl','wb'))


