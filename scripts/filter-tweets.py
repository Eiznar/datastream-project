import time
from kafka import KafkaConsumer, KafkaProducer
import json
import re
import requests
from matplotlib import pyplot as plt
from matplotlib import animation
import datetime as dt
import time

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

class RegrMagic(object):
    """Mock for function Regr_magic()
    """
    def __init__(self):
        self.x = 0
    def __call__(self):
        time.sleep(1)
        #self.x += 1
        return 1, 1

regr_magic = RegrMagic()

def frames():
    while True:
        yield regr_magic()


topic_name = 'raw-tweets'

consumer = KafkaConsumer(topic_name, bootstrap_servers=["localhost:9092"],
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id='group-1')
producer = KafkaProducer(bootstrap_servers="localhost:9092")

URL = "http://172.24.251.13:5000"
#"http://127.0.0.1:8000"

with open("archive.csv", "w+", encoding="utf8") as f :
    for message in consumer:
        #print(message.value)
        tweet = json.loads(message.value)

        # defining a params dict for the parameters to be sent to the API
        PARAMS = {'tweet' : cleanTweet(str(tweet["text"]))}
        print(PARAMS)

        # sending get request and saving the response as response object
        r = requests.get(url = URL, params = PARAMS)

        data = r.json() # {'fake': fake, 'real': real, "prediction": pred}
        #print(cleanTweet(str(tweet["text"])))
        print(data)

        f.write(str(PARAMS) + ";" + str(data['fake']) + ";" + str(data['real']) + ";" + str(data['prediction']) + "\n")




