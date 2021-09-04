#Twitter sentiment analysis
'''---> To get High Sentiment Tweet

---> To get Low Sentiment Tweet

---> To get Most Retweeted Tweet

---> To get User Producing most Tweets'''

import html
import json
import string
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt
# Collect relevant tweets through the Twitter API.
import json
import tweepy as tw

#Connecting to Twitter API
#IMPORTANT: enter proper access credential in config_twitter.py file
import config_twitter
#function to establish an initial API connection, respecting the rate limit
def connect_api_client():
    auth = tw.OAuthHandler(config_twitter.CONSUMER_KEY, config_twitter.CONSUMER_SECRET)
    auth.set_access_token(config_twitter.ACCESS_TOKEN, config_twitter.ACCESS_SECRET)
    api = tw.API(auth, wait_on_rate_limit=True)
    try:
        # returns False if credentials could not be verified
        api.verify_credentials()
        user = api.verify_credentials()
        if not user:
            raise("Credentials could not be verified: Please check config.py")
        print(f"Connected to Twitter API as {user.name}")
    except Exception as e:
        raise e
    return api
api = connect_api_client()

#Collecting Data for Charity
#construct a search query
query = 'charity OR "charity donation" OR "spend charity" OR "pay charity" -filter:retweets'
#decide how many tweets to query
ntweets = 2000
#search and collect relevant tweets
tweets = [tweet._json for tweet in tw.Cursor(api.search, q=query, lang="en", tweet_mode='extended').items(ntweets)]

# save tweets data to json file
file_out = f"raw_tweet_data_charities_{ntweets}.json"
with open(file_out, mode='w') as f:
    f.write(json.dumps(tweets, indent=2))
# First collect the data in json-file; specify file name here (adjust the number as queried)
fjson = 'raw_tweet_data_charities_2000.json'
# read json file with tweets data
with open(fjson) as file:
    data = json.load(file)

# create pandas dataframe from tweet text content
df_tweets = pd.DataFrame([t['full_text'] for t in data], columns=['text'])
# add selected columns from tweet data fields
df_tweets['retweets'] = [t['retweet_count'] for t in data]
df_tweets['favorites'] = [t['favorite_count'] for t in data]
df_tweets['user'] = [t['user']['screen_name'] for t in data]

#Preprocessing
stop_words = set(stopwords.words('english'))
def text_cleanup(s):
    s_unesc = html.unescape(re.sub(r"http\S+", "", re.sub('\n+', ' ', s)))
    s_noemoji = s_unesc.encode('ascii', 'ignore').decode('ascii')
    # normalize to lowercase and tokenize
    wt = word_tokenize(s_noemoji.lower())
    # filter word-tokens
    wt_filt = [w for w in wt if (w not in stop_words) and (w not in string.punctuation) and (w.isalnum())]
    # return clean string
    return ' '.join(wt_filt)
#add clean text column
#NOTE: apply in pandas applies a function to each element of the selected column
df_tweets['text_clean'] = df_tweets['text'].apply(text_cleanup)
df_tweets
df_tweets.to_csv (r'Charities_tweets.csv', index = False, header=True)

# sentiment analysis
def sentim_polarity(s):
    return TextBlob(s).sentiment.polarity
def sentim_subject(s):
    return TextBlob(s).sentiment.subjectivity
df_tweets['polarity'] = df_tweets['text_clean'].apply(sentim_polarity)
df_tweets['subjectivity'] = df_tweets['text_clean'].apply(sentim_subject)

#Highest sentiment tweets
df_tweets.sort_values(by='polarity', ascending=False).head(20)

# Lowest sentiment tweets
df_tweets.sort_values(by='polarity', ascending=True).head(20)

# most retweeted content
df_tweets.sort_values(by='retweets', ascending=False).head(20)

# users producing most retweeted content
df_tweets.sort_values(by='retweets', ascending=False).head(20)['user']
