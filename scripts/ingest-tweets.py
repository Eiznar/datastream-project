import tweepy
from kafka import KafkaProducer
import json 
import time

from configparser import ConfigParser

configur = ConfigParser()
configur.read('../config.ini')
api_key = configur.getint('TwitterAPI','api_key')

# client = tweepy.Client(bearer_token=config.api_key)
client = tweepy.Client(bearer_token=api_key)


# Replace with your own search query
query = 'covid -is:retweet'


# Replace the limit=1000 with the maximum number of Tweets you want

producer = KafkaProducer(bootstrap_servers="localhost:9092")
topic_name = "raw-tweets"

for tweet in tweepy.Paginator(client.search_recent_tweets, query=query, 
                              tweet_fields=['created_at', 'lang', 'possibly_sensitive'], 
                              max_results=100).flatten(limit=10000):
    # Only store english tweets
    if tweet["lang"] == "en" :
        message = json.dumps(tweet['data'],indent=4)
        producer.send(topic_name, str(message).encode('utf-8'))
        print("Sending message {} to topic: {}".format(message, topic_name))
        time.sleep(1)