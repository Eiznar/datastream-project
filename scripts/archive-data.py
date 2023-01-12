import os
import json

from kafka import KafkaConsumer

# Set up the Kafka consumer
consumer = KafkaConsumer(
    "raw-tweets", "en-tweets", "fr-tweets", "positive-tweets", "negative-tweets",
    bootstrap_servers=["localhost:9092"],
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="group-1")

# Open a file for writing the archived data
with open("archive.txt", "w") as f :
    # Continuously listen to the Kafka topics
    for message in consumer:
        # Write the data from the message to the file
        f.write(str(message.value))