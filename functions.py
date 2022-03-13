## import ##
# basic modules #
import pandas as pd
import numpy as np
from IPython.display import display
import re
import pickle
# visualization modules #
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
# NLP modules # 
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words("english"))
# ML modules
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss, f1_score

def load_data(path):
    df = pd.read_csv(path, sep=',', header=[0], encoding='utf_8')
    print('Data has been loaded')
    return df

def basic_info(df, show_info, text_field, no_rows_show, no_texts_show, print_text, print_df = False):
    if show_info == True:
        print('Dimension of the table:',df.shape)
        df['documentId'] = df['documentId'].apply(str)
        display(df.info())
        display(df.describe(include='O').transpose())
    if print_df == True:
        print('First {} rows of the table: '.format(no_rows_show))
        display(df.head(no_rows_show))
    if print_text == True:
        print('{} random text contents: '.format(no_texts_show))
        cl = df['textContent'].sample(no_texts_show).values
        for c in cl:
            print(c)
    return df

def output_variable(df, outcome_variable, show_histogram):
    print('Unique values in a column {} are :'.format(outcome_variable))
    print(df[outcome_variable].unique())
    if show_histogram == True:
        print("Distribution of the categories")
        fig = px.histogram(df, x=outcome_variable, color=outcome_variable, title='Distribution of labels')
        fig.show()      
    return df 

def drop_missing_values(df):
    print('Drop NULL values if any is present')
    if (df.isnull().sum() > 0).any() == True:
        print("show columns with NULL values:")
        null_counts = df.isnull().sum()
        print(null_counts[null_counts > 0])
        df.dropna(inplace=True)
    else:
        print("...No NULL values in the dataset")   
    return df

def add_new_label(df, outcome_variable, bi_label):
    df[bi_label] = np.select(condlist=[df[outcome_variable] != 'ok',df[outcome_variable] == 'ok'],choicelist=[0,1])
    print('New binary label has been added')
    return df

def clean_text(text: str, stemming):
    """
        - convert all whitespaces (tabs etc.) to single wspace if not for embedding (but e.g. tdf-idf)
        - remove any html tags (< /br> often found)
        - Keep only ASCII + European Chars and whitespace, no digits
        - remove single letter chars
        - all lowercase
        - remove stopwords, punctuation and stemm
    """
    RE_WSPACE = re.compile(r"\s+", re.IGNORECASE)
    RE_ASCII = re.compile(r"[^A-Za-zÀ-ž ]", re.IGNORECASE)
    RE_SINGLECHAR = re.compile(r"\b[A-Za-zÀ-ž]\b", re.IGNORECASE)
    RE_SQUAREBRACKET = re.compile(r"\[[^\]]*\]", re.IGNORECASE)

    text = re.sub(RE_ASCII, " ", text)
    text = re.sub(RE_SINGLECHAR, " ", text)
    text = re.sub(RE_WSPACE, " ", text)

    word_tokens = word_tokenize(text, language='english')
    words_tokens_lower = [word.lower() for word in word_tokens]
    
    if stemming == True:
        words_filtered = [stemmer.stem(word) for word in words_tokens_lower if word not in stop_words]
        text_prepared = " ".join(words_filtered)
    else:
        text_prepared = " ".join(word for word in words_tokens_lower)
    return text_prepared

def vectorizer(df, input_to_vec, text_cols = False):  
    vectorizer = TfidfVectorizer(analyzer="word", norm="l2")
    
    if input_to_vec == 'one_col':
        vectorizer.fit(df['text_prepared'])
        X = vectorizer.transform(df['text_prepared'])
        print('One input variable has been processed by the vectorizer')
    elif input_to_vec == 'more_cols':
        df['combined_input'] = df[text_cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        vectorizer.fit(df['combined_input'])
        X = vectorizer.transform(df['combined_input'])
        print('More input variables have been processed by the vectorizer')
        
    filename = 'pickle_vectorizor.sav'
    pickle.dump(vectorizer, open(filename, 'wb'))
    print('Vectorizor has been saved')
    return X, df

def train_model(df, X, train_size, classifier, outcome_variable, bi_label):

    if classifier == 'multi':
        print('Multiclass classifier has been selected')
        y = df[[outcome_variable]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_size, random_state = 0)
        classifiers = [
#             RandomForestClassifier(random_state=0),
            LinearSVC(max_iter=2000),
#             SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=1000, tol=1e-4),
#             MultinomialNB(),
#             LogisticRegression(random_state=0, multi_class='auto', solver='lbfgs', max_iter=2000)
        ]
    
    if classifier == 'bi':
        print('Binary classifier has been selected')
        y = df[[bi_label]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 0)
        classifiers = [
#             RandomForestClassifier(random_state=0),
#             LinearSVC(max_iter=2000),
            LogisticRegression(random_state=0, multi_class='ovr', solver='lbfgs', max_iter=2000)
        ]       
        
    #  get names of the objects in list
    names = [re.match(r"[^\(]+", name.__str__())[0] for name in classifiers]
    print(f"Classifiers to test: {names}")
    print('_'*110)
    results_ = {}
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
#         y_proba = clf.predict_proba(X_test)
#         print(y_proba)
        print(f"Training classifier: {name}")
        print(classification_report(y_test,y_pred))
        print('accuracy score: ',accuracy_score(y_test, y_pred))
    
    filename = 'finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
    print('Model has been saved')
    return clf

def save_model(model):
    # save the model to disk
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    return print('Model has been saved')

