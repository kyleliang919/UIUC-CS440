import numpy as np
import random
def squared_dist(v1,v2):
	return np.sum(np.square(np.subtract(v1,v2)))
class KM:
	def __init__(self,k,epoch):
		self.k=k
		self.epoch=epoch
		self.clusters=[[[],[]] for i in range(k)]
	def cluster(self,train_set):
		epoch=0
		self.random_initialize(train_set)
		while(epoch<self.epoch):
			self.empty_the_cluster()
			#assign data points to clusters
			for i in range(len(train_set)):
				temp=[]
				for cluster in self.clusters:
					temp.append(squared_dist(train_set[i],cluster[0]))
				self.clusters[np.argmin(temp)][1].append(i)
			#update centroids and delete the empty clusters
			new_clusters=[]
			for cluster in self.clusters:
				size=len(cluster[1])
				if size==0:
					continue
				temp=[0]*len(cluster[0])
				for idx in cluster[1]:
					temp=np.add(temp,train_set[idx])
				cluster[0]=np.divide(temp,size)
				new_clusters.append(cluster)
			self.clusters=new_clusters
			epoch+=1
	def random_initialize(self,train_set):
		for i in range(len(self.clusters)):
			self.clusters[i][0]=np.array(train_set[random.randint(0,len(train_set)-1)])
	def empty_the_cluster(self):
		for i in range(len(self.clusters)):
			self.clusters[i][1]=[]

	def ss_total(self,train_set):
		sum=0
		for cluster in self.clusters:
			for idx in cluster[1]:
				sum+=squared_dist(cluster[0],train_set[idx])
		return sum
