import preprocessor as p
from nltk.corpus import stopwords
import nltk
import string
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
table = str.maketrans('', '', string.punctuation)

def preprocessing(tweet):
    tweet = tweet.lower() 
    
    tweet = p.clean(tweet) 
    
    tokens = nltk.word_tokenize(tweet)

    tokens = [ token.translate(table) for token in tokens ]
    
    tokens = [ token.translate(table) for token in tokens]
    
    tokens = [ token for token in tokens if token not in stop_words]
    
    tokens = [stemmer.stem(token) for token in tokens]
    
    return ' '.join(tokens)