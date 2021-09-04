import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import geopandas as gp
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
from nltk.stem.porter import *
stemmer = PorterStemmer()
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
import nltk
nltk.download('movie_reviews')
nltk.download('punkt')
from textblob import TextBlob
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer

tweets = pd.read_csv('/content/Charities_tweets.csv')

#Text pre-processing
#function to remove @user
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i,'',input_txt)
    return input_txt

#additional cleaning
tweets['Tweet'] = np.vectorize(remove_pattern)(tweets['text'], '@[\w]*') # create new column with removed @user
tweets['Tweet'] = tweets['Tweet'].apply(lambda x: re.split('http:\/\/.*', str(x))[0]) # remove urls
tweets['Tweet'] = tweets['Tweet'].str.replace('[^a-zA-Z#]+',' ') # remove special characters, numbers, punctuations

#Creating a function that takes care of all the preprocessing stuff.
def preprocess():
    tweets['Tweet'] = tweets['Tweet'].str.lower() # Ensuring all words in the Tweet column of training data are lowercased
    #Parsing the stop_words.txt file and storing all the words in a list.
    stopwords = nltk.corpus.stopwords.words("english")

    #Removing all stopwords from all the tweets in training data.
    tweets["Tweet"] = tweets["Tweet"].apply(lambda func: ' '.join(sw 
                                            for sw in func.split() 
                                            if sw not in stopwords))
    #Training Data
    tweets['Tweet'] = tweets['Tweet'].str.replace(r'http?://[^\s<>"]+|www\.[^\s<>"]+', '') # Removing hyperlinks from all the tweets
    tweets['Tweet'] = tweets['Tweet'].str.replace('@[A-Za-z0-9]+', '') # Removing usernames from all the tweets.
    tweets['Tweet'] = tweets['Tweet'].str.replace(r'\B#\w*[a-zA-Z]+\w*', '') # Removing hashtags, including the text, from all the tweets
    tweets['Tweet'] = tweets['Tweet'].str.replace('\d+', '') # Removing numbers from all the tweets
    special_chars = ["!",'"',"%","&","amp","'","(",")", "*","+",",","-",".","/",":",";","<","=",">","?","[","\\","]","^","_","`","{","|","}","~","–","@","#","$"]
    for c in special_chars:
        tweets['Tweet'] = tweets['Tweet'].str.replace(c,'') # Removing all special characters from all the tweets
preprocess()

#create new variable tokenized tweet 
tokenized_tweet = tweets['Tweet'].apply(lambda x: x.split())
#remove stopwords
stopwords = nltk.corpus.stopwords.words("english")
tokenized_tweet = [w for w in tokenized_tweet if w not in stopwords]
#join tokens into one sentence
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
#change df['Tweet'] to tokenized_tweet
tweets['Tweet']  = tokenized_tweet
#tweets after cleaning
tweets['Tweet']

#Deriving sentiment
#assign sentiment scores
scores = []
for tweet in tweets['Tweet']:
    score = sia.polarity_scores(tweet)
    scores.append(score['compound'])
tweets['sentiment_scores'] = scores
tweets['sentiment_derived'] = ["positive" if w >0 else "negative" if w < 0 else "neutral" for w in tweets['sentiment_scores']]
tweets['sentiment_scores']

#percent match between assigned and derived sentiment
tweets['match'] = (tweets['sentiment_derived']==tweets['Geo_Enabled']).astype(int)
tweets[['Geo_Enabled','sentiment_derived','match']]
tweets['match'].mean()

#crosstab of assigned vs derived sentiment
pd.crosstab(tweets.Geo_Enabled, tweets.sentiment_derived)

import nltk
nltk.download('movie_reviews')
nltk.download('punkt')

blobber = Blobber(analyzer=NaiveBayesAnalyzer())
blob = TextBlob("i love it!")
print(blob.sentiment)
blob = blobber("i hate it!")
print(blob.sentiment)

scores = []
for tweet in tweets['Tweet']:
    score = TextBlob(tweet)
    scores.append(score.sentiment[0])
tweets['textblob_scores'] = scores
tweets['textblob_derived'] = ["positive" if w >0 else "negative" if w < 0 else "neutral" for w in tweets['textblob_scores']]
pd.crosstab(tweets.Geo_Enabled, tweets.textblob_derived)
pd.crosstab(tweets.sentiment_derived, tweets.textblob_derived)

#tweets.to_csv('test.csv')

def combined_sentiment(tweets):
    if (tweets['textblob_derived'] == 'negative') or (tweets['sentiment_derived'] == 'negative'):
        return 'negative'
    if (tweets['textblob_derived'] == 'neutral') and (tweets['sentiment_derived'] == 'positive'):
        return 'neutral'
    if (tweets['textblob_derived'] == 'positive') and (tweets['sentiment_derived'] == 'neutral'):
        return 'neutral'
    if (tweets['textblob_derived'] == 'neutral') and (tweets['sentiment_derived'] == 'neutral'):
        return 'negative'
    if (tweets['textblob_derived'] == 'positive') and (tweets['sentiment_derived'] == 'positive'):
        return 'positive'
    else:
        return '0'
tweets['final_derived'] = tweets.apply(combined_sentiment, axis=1)
pd.crosstab(tweets.final_derived, tweets.Geo_Enabled)
