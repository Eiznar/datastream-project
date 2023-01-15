# datastream-project

This repository contains the work of the short Twitter project for the Data Stream Processing Course in the Master 2 Data Science (2022-2023).

# How to run :

    - Download data folder from https://github.com/smkerr/COVID-fake-news-detection and place it in root directory.
    - Start Zookeeper and Kafka
    - Start the Flask app.py to be able to generate prediction of a tweet
    - Run ingest-tweets.py to start ingesting tweets
    - Run filter-tweets.py to read consumer, send http request and store every prediction in a .csv file
    - Run histogram.py to start a real time histogram with prediction score

# Predictive models

## The API

An exposed API loads a classification model that was previously trained to classify COVID tweets. The two categories are:
- Real news
- Anything that is not real news (non informative posts or fake news)


## The models
The logistic regression model is loaded, but three models are available.
- A logistic regression model (few seconds of training)
- A Support Vector Classifier (few seconds of training)
- A fine-tuned neural network (roughly 30 minutes of fine-tuning on Colab)

# Notes

The `utils.py` file contains the script of the `functions.py` file in the repository of [COVID Fake News Detection](https://github.com/smkerr/COVID-fake-news-detection) by Hannah Schweren, Marco Schildt and Steve Kerr. The repository also contains the data that was used for the training and testing of the models.

The data itself is from the official competition of [COVID-19 Fake News Detection](https://github.com/parthpatwa/covid19-fake-news-detection). The dataset and the benchmark model results can be found [here](https://arxiv.org/abs/2011.03327).

@misc{patwa2020fighting, title={Fighting an Infodemic: COVID-19 Fake News Dataset}, author={Parth Patwa and Shivam Sharma and Srinivas PYKL and Vineeth Guptha and Gitanjali Kumari and Md Shad Akhtar and Asif Ekbal and Amitava Das and Tanmoy Chakraborty}, year={2020}, eprint={2011.03327}, archivePrefix={arXiv}, primaryClass={cs.CL} }
