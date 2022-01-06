from libs import *
from modeling import *
from urllib.request import urlopen
import pandas as pd


def cleaning(self, text):
    # Remove all the special characters
    process_text = re.sub(r'\W', ' ', text)

    # remove all single characters
    process_text = re.sub(r'\s+[a-zA-Z]\s+', ' ', process_text)

    # Remove single characters from the start
    process_text = re.sub(r'\^[a-zA-Z]\s+', ' ', process_text)

    # Substituting multiple spaces with single space
    process_text = re.sub(r'\s+', ' ', process_text, flags=re.I)

    # Remove numbers
    process_text = re.sub(r'[0-9]', '', process_text)

    # Removing prefixed 'b'
    process_text = re.sub(r'\^b\s+', '', process_text)

    # lowering case
    process_text = process_text.lower()

    return process_text


def tokenization(self, text):
    tokens = word_tokenize(text)
    return tokens


def spelling_correction(self, text):
    textblob =  TextBlob(text)
    text_correct = str(textblob.correct())
    return text_correct


def lemmatizing(self, tokenized_text):
    lem = WordNetLemmatizer()
    words = []
    for word in tokenized_text:
        words.append(lem.lemmatize(word, pos='v'))

    return words


def stemming(self, tokenized_text):
    stemmer = PorterStemmer()
    words = []
    for word in tokenized_text:
        words.append(stemmer.stem(word))

    return words


def remove_emojis(self, text):
    """
    https://dataaspirant.com/nlp-text-preprocessing-techniques-implementation-python/
    Result :- string without any emojis in it
    Input :- String
    Output :- String
    """
    emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002500-\U00002BEF"  # chinese char
                                u"\U00002702-\U000027B0"
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                u"\U0001f926-\U0001f937"
                                u"\U00010000-\U0010ffff"
                                u"\u2640-\u2642"
                                u"\u2600-\u2B55"
                                u"\u200d"
                                u"\u23cf"
                                u"\u23e9"
                                u"\u231a"
                                u"\ufe0f"  # dingbats
                                u"\u3030"
                                "]+", flags=re.UNICODE)

    without_emoji = emoji_pattern.sub(r'', text)
    return without_emoji


def remove_stopwords(self, tokenized_text):
    words = []
    for word in tokenized_text:
        if word not in all_stopwords:
            words.append(word)

    return words


def text_preprocessing(self, text):
    processed_text = self.cleaning(text)
    processed_text = self.remove_emojis(processed_text)
    #processed_text = self.spelling_correction(processed_text)  # Astronomical slow!
    tokenized_text = self.tokenization(processed_text)
    #tokenized_text = self.stemming(tokenized_text)
    tokenized_text = self.lemmatizing(tokenized_text)
    tokenized_text = self.remove_stopwords(tokenized_text)
        
    return ' '.join(tokenized_text)


def create_tfidf_vectorizer(X, max_df=0.8, min_df=2, stop_words='english'):
    print('Using tfidf as vectorizer')
    tfidf = TfidfVectorizer(max_features=3000, max_df=0.8, min_df=2, stop_words=stop_words)
    X_vectorized = tfidf.fit_transform(X).toarray()
    return X_vectorized


data = pd.read_csv('spam.csv', encoding="ISO-8859-1")
X, y = data['v2'], data['v1']
X = create_tfidf_vectorizer(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
config = {
    'test_size': 0.2,
    'random_state': 123,
    'metric': 'accuracy',
    'task': 'classification',
    'time_budget': 300
}

automl_clf = get_model(X_train, y_train, **config)
y_pred = automl_clf.predict(X_test)

print(accuracy_score(y_test,y_pred))

#automl=AutoMLSearch(X_train=X_train,y_train=y_train,problem_type='binary',max_batches=1,optimize_thresholds=True)
#automl.search()

#print(automl.rankings)
#best_pipeline = automl.best_pipeline
#print(automl.describe_pipeline(automl.rankings.iloc[0]["id"]))

#scores = best_pipeline.score(X_test, y_test,  objectives=evalml.objectives.get_core_objectives('binary'))
#print(f'Accuracy : {scores["Accuracy Binary"]}')
