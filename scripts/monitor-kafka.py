import os
import time

from kafka import KafkaConsumer, TopicPartition

topics = ['en-tweets', 'fr-tweets', 'negative-tweets', 'positive-tweets', 'raw-tweets']
n = len(topics)
# Set up the Kafka consumer
consumer = KafkaConsumer(
    bootstrap_servers=["localhost:9092"],
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="group-5"
)

consumer.subscribe(topics)

CLR = "\x1B[0K"
print("BEGINNING REAL-TIME MONITORING")
print("Topic name", "Partition", "Offset", "Timestamp")
print('\n\n')
lines=0
topic_info = [""] * n

while True:

    UP = f"\x1B[{lines+1}A"
    lines=0

    for i in range(n):
        topic_info_ = ""
        topic = topics[i]

        for partition in consumer.partitions_for_topic(topic):
            tp = TopicPartition(topic, partition)
            # Something better can probably be done here using consumer.committed, but for some reason it always returns None.
            offset = consumer.end_offsets([tp])[tp]
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(offset / 1000.0))
            
            topic_info_ += f"{topic}, {partition}, {offset}, {timestamp}{CLR}\n"
            lines+=1
        # Update topic info
        topic_info[i] = topic_info_
    
    all_info = ""
    for i in range(n):
        all_info += topic_info[i]
    print(f"{UP}" + all_info)
    time.sleep(1)