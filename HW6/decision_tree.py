import math
import numpy as np
import copy
# Grade:F~0,S~1,J~2
# BMI:N~0,O~1
#PlayBall:-~0,+~1
trainset=[[1,0,0,1],[2,1,1,2],[0,1,1,3],[0,0,1,4],[2,0,0,5],[1,1,0,6],[2,0,0,7],[1,0,0,8],[2,0,0,9],[2,0,1,10],[0,0,1,11],[2,0,0,12],[1,1,1,13],[0,0,1,14],[2,1,0,15],[2,1,0,16],[0,0,1,17],[0,0,1,18],[2,1,0,19],[1,1,1,20]]
testset=[[0,0,1,21],[0,0,0,22],[2,1,1,23],[1,1,1,24],[2,0,0,25],[2,0,0,26],[1,1,1,27],[0,1,1,28],[1,0,1,29],[1,1,0,30]]
#validation of the data set
Grade=["F","S","J"]
BMI=["N","O"]
PlayBall=["-","+"]
print("training set validation")
print("Grade ","BMI ","PlayBall ")
count=1
for set in trainset:
    print(count,":",Grade[set[0]]," ",BMI[set[1]]," ",PlayBall[set[2]])
    count+=1
count=1
print("testing set validation")
print("Grade ","BMI ","PlayBall")
for set in testset:
    print(count,":",Grade[set[0]]," ",BMI[set[1]]," ",PlayBall[set[2]])
    count+=1

def entropy(samples,label):
    size=len(samples)
    num_each_label={}
    for sample in samples:
        if sample[label] in num_each_label:
            num_each_label[sample[label]]+=1
        else:
            num_each_label[sample[label]]=1
    probs=[]
    for key,value in num_each_label.items():
        probs.append(value/size)
    entropy=0
    for p in probs:
        if p==0:
            continue
        else:
            entropy+=-p*math.log(p)
    return entropy
def homogeneous(sample_set,label):
    temp=sample_set[0][label]
    for sample in sample_set:
        if sample[label]!=temp:
            return False
    return True
def majorLabel(sample_set,label):
    major=0
    count=1
    labels={}
    for sample in sample_set:
        if sample[label] in labels:
            labels[sample[label]]+=1
        else:
            labels[sample[label]]=1
    for key,value in labels.items():
        if count==1:
            major=key
            count=0
        if labels[major]<value:
            major=key
    return major
class DTNode(object):
    def __init__(self,test=None,isLeaf=False,label=None):
        self.children={}
        self.test=test
        self.isLeaf=isLeaf
        self.label=label
class decision_tree(object):
    def __init__(self,label,features):
        self.root=DTNode()
        self.features=features# 0~GRADE, 1~BMI
        self.label=label
    def growTree(self,node,trainset,features):
        print("features***********",features)
        if len(features)==0 or homogeneous(trainset,self.label):
            print("Training sample in this leaf")
            for sample in trainset:
                print(sample[3])
            leaf_label=majorLabel(trainset,self.label)
            node.isLeaf=True
            node.label=leaf_label
            print("label at this leaf:",leaf_label)
            return None
        ES=entropy(trainset,self.label)
        size=len(trainset)
        information_gains=[]
        for feature in features:
            Sv={}
            for sample in trainset:
                if sample[feature] in Sv:
                    Sv[sample[feature]].append(sample)
                else:
                    Sv[sample[feature]]=[sample]
            gain=ES
            for key,value in Sv.items():
                gain-=len(value)/size*entropy(value,self.label)
            information_gains.append(gain)
        idx=np.argmax(information_gains)
        feature=features[idx]
        features.remove(features[idx])
        print("test chosen:",feature,"\ninformation gain:",information_gains[idx])
        Sv={}
        node.test=feature
        for sample in trainset:
            if sample[feature] in Sv:
                Sv[sample[feature]].append(sample)
            else:
                Sv[sample[feature]]=[sample]
        for key,value in Sv.items():
            print("test:",feature,"feature=",key)
            node.children[key]=DTNode()
            self.growTree(node.children[key],Sv[key],copy.copy(features))
    def test(self,label,testset):
        print("Start testing:")
        for sample in testset:
            pred=self.predict(sample)
            #print("sample idx:",sample[-1],";","prediction: ",pred,";","real label: ",sample[label])
            print(sample[-1],":",pred)
    def predict(self,sample):
        temp=self.root
        while temp.isLeaf!=True:
            temp=temp.children[sample[temp.test]]
        pred=temp.label    
        return pred
DT=decision_tree(0,[1,2])
DT.growTree(DT.root,trainset+testset,DT.features)
#testset=trainset[10:19]
DT.test(0,trainset+testset)
        