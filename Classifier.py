import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from nltk.tokenize.treebank import TreebankWordDetokenizer

df = pd.read_csv('Articles.csv')
lem = WordNetLemmatizer()
stemmer = PorterStemmer()
tokenizer = nltk.RegexpTokenizer(r"\w+")
stop_words = stopwords.words('english')

# eliminate stop words in the articles
df['Summary'] = df['Summary'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

articles = df['Summary'].values.tolist()
classes = df['Classification'].values.tolist()

# tokenize
for i in range(len(articles)):
    articles[i] = tokenizer.tokenize(articles[i])

# clean
for i in range(len(articles)):
    for j in range(len(articles[i])):
        articles[i][j] = articles[i][j].lower()
        articles[i][j] = stemmer.stem(articles[i][j])
        articles[i][j] = lem.lemmatize(articles[i][j])
        
# bag of words representation for every article
BoW = []
BoW_dict = {}
for i in range(len(articles)):
    BoW.append(BoW_dict.copy())

for i in range(len(articles)):
    for word in articles[i]:
        if word not in BoW[i].keys():
            BoW[i][word] = 1
        else:
            BoW[i][word] += 1

# detokenize the articles because I needed to reshape the data
for i in range(len(articles)):
    articles[i] = TreebankWordDetokenizer().detokenize(articles[i])

articles = pd.factorize(articles)[0].reshape(-1, 1)
classes = pd.factorize(classes)[0].reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(articles, classes, test_size=0.3,random_state=42)
classifier = GaussianNB()
classifier.fit(X_train, y_train)
predict_test = classifier.predict(X_test)

print("\nPrediction Accuracy:", metrics.accuracy_score(y_test, predict_test))

