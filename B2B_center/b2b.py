import nltk
import pymorphy2
import re
#nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.metrics import accuracy_score


# M A I N
df_train = pd.read_csv("train.csv", usecols = [1, 2])
df_test = pd.read_csv("test.csv", usecols = [1, 2])
morph = pymorphy2.MorphAnalyzer()
count_vectorizer = CountVectorizer()
#print(df_train.head())
raw_text = list(df_train['proc_name'])
split = len(raw_text)
raw_text = raw_text[ : int(split/50)]
#raw_text += list(df_test['proc_name'])
stop_words = set(stopwords.words("russian"))
pattern = r"[^\w]"
pattern_num = r"[^\D]"
text_new = []
#print(raw_text)
for text in raw_text:
    text = re.sub(pattern, " ", text)
    text = re.sub(pattern_num, " ", text)
    text = text.replace("_", " ")
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        sentence = sentence.lower()
        #print(sentence)
        words = nltk.word_tokenize(sentence)
        #print(words)
        words_new = []
        for word in words:
            word_new = morph.parse(word)[0].normal_form
            words_new.append(word_new)
        words_new = [wrd for wrd in words_new if wrd not in stop_words]
        words_new = list(set(words_new))
        words_new_str = ""
        for wrd in words_new:
            words_new_str = words_new_str + " " + wrd
        #print(words_new_str)
    text_new.append(words_new_str)
print("Text cleared")
print(text_new)
word_features = count_vectorizer.fit_transform(text_new).toarray()
word_columns = count_vectorizer.get_feature_names()
df_train_processed = pd.DataFrame(word_features, columns = word_columns)
print(df_train_processed.head())

X_train = word_features[ : split]
y_train = np.array(list(df_train['target'])[ : int(split/50)])

model = KNeighborsClassifier(n_jobs = -1)
cv_gen = ShuffleSplit(n_splits = 10, test_size = 0.25, random_state = 0)
model_gs = GridSearchCV(model,
                        {
                            'n_neighbors': [2, 4, 6, 8],
                            'weights': ['uniform', 'distance']
                        },
                        scoring = 'accuracy',
                        n_jobs = -1,
                        cv = cv_gen
                        )
model_gs.fit(X_train, y_train)
print(model_gs.best_params_)
print("Accuracy score", model_gs.best_score_)
