# datastream-project

This repository contains the work of the short Twitter project for the Data Stream Processing Course in the Master 2 Data Science (2022-2023).

# How to train a classification model :
    - Download the training data from the repository of [COVID Fake News Detection](https://github.com/smkerr/COVID-fake-news-detection) by Hannah Schweren, Marco Schildt and Steve Kerr. The data should be in a `data` folder at the most parent directory in this repository.
    - Run either `logreg.py` (Logistic Regression) or `svm.py` (SVM Classifier) in order to train a classifier on the training data.
    - A model is saved as a `.pkl` file that can be loaded while running the application file.

# How to run :

    - Start Zookeeper and Kafka
    - Complete the config.ini with your bearer_token in 'api_key' section. Port is set automatically.
    - Launch the Flask app by running the `app.py`file.
    - Then run `ingest-tweets.py` to start ingesting tweets
    - Then run `filter-tweets.py` to read consumer, send http request and store every prediction in a .csv file
    - Finally run `histogram.py` to start a real time histogram with prediction score

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
