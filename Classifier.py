import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('Articles.csv')
lem = WordNetLemmatizer()
stemmer = PorterStemmer()
tokenizer = nltk.RegexpTokenizer(r"\w+")
stop_words = stopwords.words('english')

# eliminate stop words in the articles
df['Summary'] = df['Summary'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
articles = df.values.tolist()

# tokenize
for article in articles:
    article[0] = tokenizer.tokenize(article[0])

# clean
for i in range(len(articles)):
    for j in range(len(articles[i][0])):
            articles[i][0][j] = articles[i][0][j].lower()
            articles[i][0][j] = stemmer.stem(articles[i][0][j])
            articles[i][0][j] = lem.lemmatize(articles[i][0][j])

# bag of words representation for every article
BoW = []
BoW_dict = {}
for i in range(len(articles)):
    BoW.append(BoW_dict.copy())

for i in range(len(articles)):
    for word in articles[i][0]:
        if word not in BoW[i].keys():
            BoW[i][word] = 1
        else:
            BoW[i][word] += 1


print(BoW[0])


