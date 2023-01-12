import os
import json

from kafka import KafkaConsumer
import re

def cleanTweet(tweet: str) -> str:
    tweet = re.sub(r'http\S+', '', str(tweet))
    tweet = re.sub(r'bit.ly/\S+', '', str(tweet))
    tweet = tweet.strip('[link]')

    # remove users
    tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', str(tweet))
    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', str(tweet))

    # remove punctuation
    my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@â'
    tweet = re.sub('[' + my_punctuation + ']+', ' ', str(tweet))

    # remove number
    tweet = re.sub('([0-9]+)', '', str(tweet))

    # remove hashtag
    tweet = re.sub('(#[A-Za-z]+[A-Za-z0-9-_]+)', '', str(tweet))

    return tweet


# Set up the Kafka consumer
consumer = KafkaConsumer(
    "raw-tweets", "en-tweets", "fr-tweets", "positive-tweets", "negative-tweets",
    bootstrap_servers=["localhost:9092"],
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="group-1")


# Open a file for writing the archived data
with open("archive.csv", "a+") as f :
    # Continuously listen to the Kafka topics
    for message in consumer:
        # Write the data from the message to the file
        tweet = json.loads(message.value)
        f.write(str(tweet["id"]) + ";" + cleanTweet(str(tweet["text"])) + "\n")
