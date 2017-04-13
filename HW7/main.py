import numpy as np
from multiclass_perceptron import Multiclass_Perceptron
#from perceptron import Perceptron
import pickle

train_set=pickle.load(open("hw7-data/train.p","rb"))
test_set=pickle.load(open("hw7-data/test.p","rb"))


multi_perc=Multiclass_Perceptron(784,25,0.001,10,True)
#multi_perc.tuneParameters(train_set,5)
multi_perc.train(train_set)
print("accuracy on the test_set")
multi_perc.test(test_set)
print("accuracy on the train_set")
multi_perc.test(train_set)
"""
perc=Perceptron(784,1,0.01,0)
perc.train(train_set)
#print(perc.w)
print(perc.test(test_set))
"""