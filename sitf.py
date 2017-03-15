#!/Users/FWerpers/.virtualenvs/sci-kit/bin/python

""" This script is used to create classifiers
"""

from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.pipeline import Pipeline
from sklearn.pipeline import _name_estimators
from sklearn.preprocessing import Normalizer

from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_curve

from sklearn.externals import joblib

from nltk.stem import SnowballStemmer

import numpy as np
import re
import csv
import time

# import matplotlib
# matplotlib.use('TkAgg')
# from matplotlib import pyplot as plt

# Seed
#SEED = int(time.clock()*1e6)
SEED = 42

# Save paths
SAVE_PATH_EN = 'classifiers/en/'
SAVE_PATH_SV = 'classifiers/sv/'

# Corpus paths
CORPUS_PATH_EN = 'data/tweets_en.csv'
CORPUS_PATH_SV = 'data/tweets_sv.csv'

# Label paths
LABELS_PATH_EN = 'data/labels_en.csv'
LABELS_PATH_SV = 'data/labels_sv.csv'

# Stopword paths
STOPWORDS_PATH_EN = 'data/stopwords_en.txt'
STOPWORDS_PATH_SV = 'data/stopwords_sv.txt'

# Token pattern
REG_INCLUDE = [
	r'(?:\#+[\w_]+[\w\'_\-]*[\w_]+)', # hash-tags
	r'(?:[a-zA-ZåäöÅÄÖ_]+\'(?:[std]|ll))',
	r'(?:[a-zA-ZåäöÅÄÖ_]+)'
]

TOKEN_PATTERN_STR = r'(' + '|'.join(REG_INCLUDE) + ')'
TOKEN_PATTERN = re.compile(TOKEN_PATTERN_STR, re.VERBOSE | re.IGNORECASE)

# Remove pattern
REG_REMOVE = [
	r'(?:@[\w_]+)', # @-mentions
	r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+' # URLs
	#r'[0-9]+' # numbers
]

REMOVE_PATTERN_STR = r'(' + '|'.join(REG_REMOVE) + ')'
REMOVE_PATTERN = re.compile(REMOVE_PATTERN_STR, re.VERBOSE | re.IGNORECASE)

# Football result pattern
REG_SCORE = r'(?:\d-\d)'
SCORE_PATTERN = re.compile(REG_SCORE)

class bc:
	RED = '\033[91m'
	GREEN = '\033[92m'
	YELLOW = '\033[93m'
	BLUE = '\033[94m'
	MAGENTA = '\033[95m'
	CYAN = '\033[96m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'

	COLORS = [
		BLUE,
		MAGENTA,
		CYAN,
		RED,
		GREEN,
		YELLOW
	]

class Lang():

	def __init__(self, lang):
		if lang=='sv':
			self.stemmer = SnowballStemmer('swedish')
			with open(STOPWORDS_PATH_SV, 'r') as f:
				self.stopwords = f.read().splitlines()
		elif lang=='en':
			self.stemmer = SnowballStemmer('english')
			with open(STOPWORDS_PATH_EN, 'r') as f:
				self.stopwords = f.read().splitlines()
		else:
			raise ValueError('Language "' + lang + '" not supported.')

	def preprocessor(self, tweet):
		tweet = REMOVE_PATTERN.sub('', tweet).lower()
		tweet = SCORE_PATTERN.sub('SCORE', tweet)
		return tweet

	def tokenizer(self, tweet):
		tokens = TOKEN_PATTERN.findall(tweet)
		tokens = [self.stemmer.stem(token) for token in tokens]
		return tokens

class TweetHashingVectorizer(HashingVectorizer):
	def __init__(self, lang, binary=False):

		self.lang = lang

		super().__init__(
			input='content',
			lowercase=True,
			stop_words=self.lang.stopwords,
			preprocessor=self.lang.preprocessor,
			tokenizer=self.lang.tokenizer,
			binary=binary)

class TweetClassifyingPipeline(Pipeline):
	def __init__(self, lang):
		self.lang = lang

		self.vectorizer = TweetHashingVectorizer(self.lang)
		self.clf = SGDClassifier(
			alpha=1e-3,
			random_state=SEED)

		pipeline_steps = [('vect', self.vectorizer), ('clf', self.clf)]
		super().__init__(pipeline_steps)

	def grid_search(self, X, y, scoring):

		param_grid = {
			'vect__binary': (True, False),
			'clf__loss': ('hinge', 'log'),
			'clf__penalty': ('none', 'l2'),
			'clf__class_weight': (None, 'balanced')
		}

		gs = GridSearchCV(self, param_grid, scoring=scoring, cv=10)
		gs.fit(X,y)

		return gs

	def partial_fit(self, tweets, labels):
		X = self.vectorizer.transform(tweets)
		y = labels
		self.clf.partial_fit(X, y)

	def save(self, path):
		joblib.dump(self, path)

def get_data(corpus_path, labels_path, soccer_as_positive=False):
	"""Get tweets and labels from CSV files"""

	# Get tweets
	with open(corpus_path, 'r') as corpus:
		reader = csv.reader(corpus, dialect='excel')
		tweets = [row[0] for row in reader]

	X = tweets

	# Get labels
	with open(labels_path, 'r') as labels:
		reader = csv.reader(labels, dialect='excel')
		y = [int(row[0]) for row in reader]

	y = np.array(y)

	# Revert labels since we want emphasis on non-football tweets
	# (Non-football tweets should be 1 and the others 0)
	if not soccer_as_positive:
		y ^= 1

	return (X,y)

def create_classifier(lang_code, pickle=False):

	lang = Lang(lang_code)
	if lang_code == 'sv':
		corpus_path = CORPUS_PATH_SV
		labels_path = LABELS_PATH_SV
		save_path = SAVE_PATH_SV
	elif lang_code == 'en':
		corpus_path = CORPUS_PATH_EN
		labels_path = LABELS_PATH_EN
		save_path = SAVE_PATH_EN

	(X,y) = get_data(corpus_path, labels_path, soccer_as_positive=False)
	print()
	print('%.2f%% about football \n' % ((1-np.mean(y))*100))
	clf = TweetClassifyingPipeline(lang)

	scoring = 'recall'

	gs = clf.grid_search(X, y, scoring)

	clf = gs.best_estimator_

	print('Grid search using \'%s\' scoring:' % scoring)
	print(gs.best_params_)
	print('Best score: %.4f' % gs.best_score_)

	if pickle:
		clf.save(save_path + 'clf.pkl')

if __name__ == '__main__':

	create_classifier('sv', pickle=True)
	create_classifier('en', pickle=True)
