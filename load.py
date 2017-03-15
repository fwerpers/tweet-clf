#!/Users/FWerpers/.virtualenvs/sci-kit/bin/python

from sitf import Lang, TweetHashingVectorizer, TweetClassifyingPipeline
from sklearn.externals import joblib
import os

CLF_PATH = 'classifiers'

def load_classifier(lang_code):
	path = os.path.join(CLF_PATH, lang_code, 'clf.pkl')
	return joblib.load(path)

def partial_train(lang_code, tweets, labels):
	"""Train classifier with new samples
	labels should be 1 for non-football tweets"""

	clf = load_classifier(lang_code)
	clf.partial_fit(tweets, labels)

	path = os.path.join(CLF_PATH, lang_code, 'clf.pkl')
	clf.save(path)

if __name__ == '__main__':
	clf = load_classifier('sv')
	print(clf.predict(['handlar det här om fotboll?']))

	# clf.partial_fit(['handlar det här om fotboll?'], [0])

	# path = os.path.join(CLF_PATH, 'sv', 'clf.pkl')
	# clf.save(path)
