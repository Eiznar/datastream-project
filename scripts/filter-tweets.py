import time
from kafka import KafkaConsumer, KafkaProducer
import json

topic_name = 'raw-tweets'
en_topic = 'en-tweets'
fr_topic = 'fr-tweets'
consumer = KafkaConsumer(topic_name, bootstrap_servers=["localhost:9092"],
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id='group-1')
producer = KafkaProducer(bootstrap_servers="localhost:9092")


for message in consumer:
    print(message.value)
    dict = json.loads(message.value)
    if dict["lang"] == "en":
        # Write to fr-topics
        producer.send(topic=en_topic, value=message.value)
    elif dict["lang"] == "fr":
        # Write to en-topics
        producer.send(topic=fr_topic, value=message.value)
    else:
        pass