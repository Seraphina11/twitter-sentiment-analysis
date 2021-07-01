# Twitter-Sentiment-Analysis

It is a Natural Language Processing Problem where Sentiment Analysis is done by Classifying the Positive tweets from negative tweets by machine learning models for classification, text mining, text analysis, data analysis and data visualization.


## Understanding the Problem Statement

Let’s go through the problem statement once as it is very crucial to understand the objective before working on the dataset. The problem statement is as follows:

The objective of this task is to detect hate speech in tweets. For the sake of simplicity, we say a tweet contains hate speech if it has a racist or sexist sentiment associated with it. So, the task is to classify racist or sexist tweets from other tweets.

Formally, given a training sample of tweets and labels, where label ‘1’ denotes the tweet is racist/sexist and label ‘0’ denotes the tweet is not racist/sexist, your objective is to predict the labels on the given test dataset.

Note: The evaluation metric from this practice problem is F1-Score.

Take a look at the pictures below depicting two scenarios of an office space – one is untidy and the other is clean and organized.


## Dataset Information

We use and compare various different methods for sentiment analysis on tweets (a binary classification problem). The training dataset is expected to be a csv file of type `tweet_id,sentiment,tweet` where the `tweet_id` is a unique integer identifying the tweet, `sentiment` is either `1` (positive) or `0` (negative), and `tweet` is the tweet enclosed in `""`. Similarly, the test dataset is a csv file of type `tweet_id,tweet`. Please note that csv headers are not expected and should be removed from the training and test datasets.  

## Requirements

There are some general library requirements for the project and some which are specific to individual methods. The general requirements are as follows.  
* `numpy`
* `scikit-learn`
* `scipy`
* `nltk`

The library requirements specific to some methods are:
* `keras` with `TensorFlow` backend for Logistic Regression, MLP, RNN (LSTM), and CNN.
* `xgboost` for XGBoost.

**Note**: It is recommended to use Anaconda distribution of Python.

## Usage

### Preprocessing 

1. Run `preprocess.py <raw-csv-path>` on both train and test data. This will generate a preprocessed version of the dataset.
2. Run `stats.py <preprocessed-csv-path>` where `<preprocessed-csv-path>` is the path of csv generated from `preprocess.py`. This gives general statistical information about the dataset and will two pickle files which are the frequency distribution of unigrams and bigrams in the training dataset. 

After the above steps, you should have four files in total: `<preprocessed-train-csv>`, `<preprocessed-test-csv>`, `<freqdist>`, and `<freqdist-bi>` which are preprocessed train dataset, preprocessed test dataset, frequency distribution of unigrams and frequency distribution of bigrams respectively.

For all the methods that follow, change the values of `TRAIN_PROCESSED_FILE`, `TEST_PROCESSED_FILE`, `FREQ_DIST_FILE`, and `BI_FREQ_DIST_FILE` to your own paths in the respective files. Wherever applicable, values of `USE_BIGRAMS` and `FEAT_TYPE` can be changed to obtain results using different types of features as described in report.

### Baseline
3. Run `baseline.py`. With `TRAIN = True` it will show the accuracy results on training dataset.

### Naive Bayes
4. Run `naivebayes.py`. With `TRAIN = True` it will show the accuracy results on 10% validation dataset.

### Maximum Entropy
5. Run `logistic.py` to run logistic regression model OR run `maxent-nltk.py <>` to run MaxEnt model of NLTK. With `TRAIN = True` it will show the accuracy results on 10% validation dataset.

### Decision Tree
6. Run `decisiontree.py`. With `TRAIN = True` it will show the accuracy results on 10% validation dataset.

### Random Forest
7. Run `randomforest.py`. With `TRAIN = True` it will show the accuracy results on 10% validation dataset.

### XGBoost
8. Run `xgboost.py`. With `TRAIN = True` it will show the accuracy results on 10% validation dataset.

### SVM
9. Run `svm.py`. With `TRAIN = True` it will show the accuracy results on 10% validation dataset.

### Multi-Layer Perceptron
10. Run `neuralnet.py`. Will validate using 10% data and save the best model to `best_mlp_model.h5`.

### Reccurent Neural Networks
11. Run `lstm.py`. Will validate using 10% data and save models for each epock in `./models/`. (Please make sure this directory exists before running `lstm.py`).

### Convolutional Neural Networks
12. Run `cnn.py`. This will run the 4-Conv-NN (4 conv layers neural network) model as described in the report. To run other versions of CNN, just comment or remove the lines where Conv layers are added. Will validate using 10% data and save models for each epoch in `./models/`. (Please make sure this directory exists before running `cnn.py`). 

### Majority Vote Ensemble
13. To extract penultimate layer features for the training dataset, run `extract-cnn-feats.py <saved-model>`. This will generate 3 files, `train-feats.npy`, `train-labels.txt` and `test-feats.npy`.
14. Run `cnn-feats-svm.py` which uses files from the previous step to perform SVM classification on features extracted from CNN model.
15. Place all prediction CSV files for which you want to take majority vote in `./results/` and run `majority-voting.py`. This will generate `majority-voting.csv`.

## Information about other files

* `dataset/positive-words.txt`: List of positive words.
* `dataset/negative-words.txt`: List of negative words.
* `dataset/glove-seeds.txt`: GloVe words vectors from StanfordNLP which match our dataset for seeding word embeddings.
* `Plots.ipynb`: IPython notebook used to generate plots present in report.


## End Notes

In this article, we learned how to approach a sentiment analysis problem. We started with preprocessing and exploration of data. Then we extracted features from the cleaned text using Bag-of-Words and TF-IDF. Finally, we were able to build a couple of models using both the feature sets to classify the tweets.
