import tensorflow as tf
import numpy as np
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
train_set=pickle.load(open("hw7-data/train.p","rb"))
test_set=pickle.load(open("hw7-data/test.p","rb"))
feature_columns=[tf.contrib.layers.real_valued_column("",dimension=784)]
classifier=tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[1000],n_classes=10,activation_fn=tf.sigmoid)
def get_train_set():
	x=tf.constant([train_set[i][0] for i in range(len(train_set))])
	y=tf.constant([train_set[i][1] for i in range(len(train_set))])
	return x,y
def get_test_set():
	x=tf.constant([test_set[i][0] for i in range(len(test_set))])
	y=tf.constant([test_set[i][1] for i in range(len(test_set))])
	return x,y
classifier.fit(input_fn=get_train_set,steps=1000)
accuracy_score=classifier.evaluate(input_fn=get_test_set,steps=1)["accuracy"]
print("\nTest Accuracy:{0:f}\n".format(accuracy_score))
accuracy_score=classifier.evaluate(input_fn=get_train_set,steps=1)["accuracy"]
print("\nTrain Accuracy:{0:f}\n".format(accuracy_score))