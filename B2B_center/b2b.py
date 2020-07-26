import nltk
import pymorphy2
import re
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score

def TextProcessing(df, pop_words = None):
    targets = list(set(df['target']))
    #print(targets)
    targets_percentage = dict.fromkeys(targets, 0)
    for val in list(df['target']):
        targets_percentage[val] += 1
    for key in targets_percentage:
        targets_percentage[key] = targets_percentage[key]*100/len(list(df['target']))
    print("Процент каждого класса в обучающей выбокре", targets_percentage)
    '''
    Выборка крайне неравномерная
    '''
    balanced_df_train = df.loc[df['target'] == 0][ : 85]
    #print(balanced_df_train)
    for trgt in targets[1 : ]:
        balanced_df_train = pd.concat([balanced_df_train, df.loc[df['target'] == trgt][ : 85]])
    print("Кол-во строк в новом сбалансированном наборе:", len(list(balanced_df_train['target'])))

    morph = pymorphy2.MorphAnalyzer()
    count_vectorizer = CountVectorizer()
    lemmatizer = WordNetLemmatizer()
    raw_text = list(balanced_df_train['proc_name'])
    split = len(raw_text)
    raw_text = raw_text[ : int(split/1)]
    stop_words = set(stopwords.words("russian"))
    pattern = r"[^\w]"
    pattern_num = r"[^\D]"
    pattern_eng = r"['a'-'b'-'c'-'d'-'e'-'f'-'g'-'h'-'i'-'j'-'k'-'l'-'n'-'m'-'o'-'p'-'q'-'r'-'s'-'t'-'u'-'v'-'w'-'x'-'y'-'z']"
    text_new = []
    all_words = []
    for text in raw_text:
        text = text.lower()
        text = re.sub(pattern, " ", text)
        text = re.sub(pattern_num, " ", text)
        text = re.sub(pattern_eng, " ", text)
        text = text.replace("_", " ")
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            words_new = []
            for word in words:
                word_new = morph.parse(word)[0].normal_form
                word_new = lemmatizer.lemmatize(word_new, wordnet.ADJ)
                words_new.append(word_new)
            words_new = [wrd for wrd in words_new if wrd not in stop_words and len(wrd) > 3]
            all_words += words_new
            #words_new = list(set(words_new))
            words_new_str = ""
            for wrd in words_new:
                words_new_str = words_new_str + " " + wrd
        text_new.append(words_new_str)
    print("Препроцессинг текста завершен")
    all_words = list(nltk.FreqDist(all_words).keys())[ : 9071]
    if pop_words is not None:
        all_words = pop_words

    text_new_popular = []
    for sentence in text_new:
        words = nltk.word_tokenize(sentence)
        sentence_popular = ""
        for word in words:
            if word in all_words:
                sentence_popular = sentence_popular + " " + word
        text_new_popular.append(sentence_popular)

    word_features = count_vectorizer.fit_transform(text_new_popular).toarray()
    word_columns = count_vectorizer.get_feature_names()
    df_train_processed = pd.DataFrame(word_features, columns = word_columns)
    #print(df_train_processed.head())

    X = word_features[ : split]
    y = np.array(list(balanced_df_train['target'])[ : int(split/1)])
    return X, y, all_words

# M A I N
df_train = pd.read_csv("train.csv", usecols = [1, 2])
df_test = pd.read_csv("test.csv", usecols = [1, 2])

X, y, popular_words = TextProcessing(df_train)

cv_gen = ShuffleSplit(n_splits = 9, test_size = 0.7, random_state = 0)
model = RandomForestClassifier(n_estimators = 350, criterion = 'gini', max_features = 'sqrt', n_jobs = -1, random_state = 0)
'''
cv_gen = ShuffleSplit(n_splits = 10, test_size = 0.25, random_state = 0)
model_gs = GridSearchCV(model,
                        {
                            'n_estimators': [350],
                            'criterion': ['gini'],
                            'max_features': ['sqrt']
                        },
                        scoring = 'accuracy',
                        n_jobs = -1,
                        cv = cv_gen
                        )
model_gs.fit(X_train, y_train)
print(model_gs.best_params_)
print("Accuracy score", model_gs.best_score_)
'''
'''
min_samples = min(X_train.shape[0], X_test.shape[0])
X_train = X_train[ : min_samples]
X_test = X_test[ : min_samples]
y_train = y_train[ : min_samples]
y_test = y_test[ : min_samples]
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

print(X_train.shape[1])

model.fit(X_train, y_train)
#print(y_train.shape)
#print(y_test.shape)
#y_test = y_test.reshape(1, y_test.shape[0])
predicted = model.predict(X_test)
#print(predicted)
print("Точность классификации на тестовой выборке accuracy:", accuracy_score(y_test, predicted))

# В коде ниже я попытался подавать модели тестовую выборку частями, так как 100к документов физически не помещаются в оперативку.
# Но такой подход не увенчался успехом, потому что в конечном итоге каждое слово - это признак, а каждой части будет разный набор слов,
# поэтому после его преобразования в целочисленную матрицу размерность по столбцам не равно той, на которой обучена модель.
'''
model.fit(X_train, y_train)

test_classification_file = open("test_classification.txt", "w")
test_classification = []

for i in range(0, len(list(df_test['proc_name'])) - 10000, 10000):
    X, y, tmp_words = TextProcessing(df_train.iloc[i : i + 10000], popular_words)
    predicted = list(model.predict(X))
    test_classification.append(predicted)

for val in test_classification:
    test_classification_file.write(str(val) + "\n")

test_classification_file.close()
'''
