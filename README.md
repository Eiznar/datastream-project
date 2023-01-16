# datastream-project

Download data folder from https://github.com/smkerr/COVID-fake-news-detection and place it in root directory.

# How to run :

    - Start Zookeeper and Kafka
    - Run ingest-tweets.py to start ingesting tweets
    - Run filter-tweets.py to read consumer, send http request and store every prediction in a .csv file
    - Run histogram.py to start a real time histogram with prediction score
