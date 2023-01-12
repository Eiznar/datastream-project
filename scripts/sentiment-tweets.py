import os
import json
import nltk
nltk.download('vader_lexicon')
from kafka import KafkaConsumer, KafkaProducer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from vaderSentiment_fr.vaderSentiment import SentimentIntensityAnalyzer as SentimentIntensityAnalyzer_fr

# Set up the Kafka consumer
consumer = KafkaConsumer(
    "en-tweets", "fr-tweets",
    bootstrap_servers=["localhost:9092"],
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="group-1"
)

# Set up the Kafka producer
producer = KafkaProducer(
    bootstrap_servers=["localhost:9092"],
    value_serializer=lambda x: json.dumps(x).encode("utf-8")
)

# Initialize the sentiment analysis model
model_en = SentimentIntensityAnalyzer()
model_fr = SentimentIntensityAnalyzer_fr()

# Continuously listen to the Kafka topics
for message in consumer:
    # Parse the tweet from the message
    tweet = json.loads(message.value)

    # Write the tweet to the appropriate Kafka topic
    if tweet['lang'] == 'en' :
        sentiment_en = model_fr.polarity_scores(tweet["text"])
        print("Sentiment_en : {}".format(sentiment_en))
        if sentiment_en["compound"] > 0 :
            producer.send("positive-tweets", tweet)
        else:
            producer.send("negative-tweets", tweet)
    
    if tweet['lang'] == 'fr':
        sentiment_fr = model_fr.polarity_scores(tweet["text"])
        print("Sentiment_fr : {}".format(sentiment_fr))
        if sentiment_fr["compound"] > 0 :
            producer.send("positive-tweets", tweet)
        else:
            producer.send("negative-tweets", tweet)