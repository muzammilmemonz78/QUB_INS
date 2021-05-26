import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from kmodes.kmodes import KModes
import numpy as np
from sklearn.cluster import KMeans
import scipy
from scipy.cluster import hierarchy
from IPython.display import display

#load csv
data = pd.read_csv("QUB_Insurance_Data_Assignment_Training_clean.csv")

#Filter Negative Values
data = data[(data['MotorValue']>0)]
data = data[(data['Age']>0)]
#print(data.describe())


#Age Category
#data['age_bin'] = pd.cut(data['Age'], [0,30, 40, 50, 60, 70, 80, 90, 100], labels=['18-30', '30-40', '40-50','50-60','60-70','70-80', '80-90','90-100'])
data  = data.drop(['Title','GivenName','MiddleInitial','Surname','CustomerID'],axis = 1)
data_ori = data
columns_to_encode = ['CreditCardType','Occupation','Gender','Location','MotorInsurance','MotorType','HealthInsurance','HealthType','HealthDependentsAdults','HealthDependentsKids','TravelInsurance','TravelType','PrefChannel']
columns_to_scale = ['Age','MotorValue']
#Encoding
data = data.astype(str)
encoders = {column: preprocessing.LabelEncoder() for column in columns_to_encode}
for column in columns_to_encode:
    data[column] = encoders[column].fit_transform(data[column])

scaler = StandardScaler()
scaled_columns  = scaler.fit_transform(data[columns_to_scale]) 
data['Age'] = scaled_columns[:,0]
data['MotorValue'] = scaled_columns[:,1]
print(data.head())
data.to_csv("Data_Encoded.csv", sep=',')


#Correlation Matrix
# corr_mat = data.corr()
# sns.heatmap(corr_mat, annot = True)
# plt.show()

#Finding optimal number of clusters using Dendrogram
dendro=hierarchy.dendrogram(hierarchy.linkage(data,method='ward'))

wcss=[]
for i in range(1,8):
    kmeans=KMeans(n_clusters=i,init='k-means++',)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,8),wcss)
plt.show()

#Model implementation
kmeans_1=KMeans(n_clusters=4)
kmeans_1.fit(data)
cluster_pred=kmeans_1.predict(data)
cluster_pred_2=kmeans_1.labels_
cluster_center=kmeans_1.cluster_centers_
data['cluster_label'] = cluster_pred_2
print(data.groupby('cluster_label').mean())
data.to_csv("Data_Encoded_labelled.csv", sep=',')

data.groupby('cluster_label').mean().to_csv("Clusters_Summary.csv", sep=',')

data=np.array(data)

# Visualising the clusters
plt.figure(figsize=(10,8))
plt.scatter(data[cluster_pred==0,0],data[cluster_pred==0,1], s = 100, c = 'red', label ='cluster 1' )
plt.scatter(data[cluster_pred==1,0],data[cluster_pred==1,1], s = 100, c = 'blue', label ='cluster 2' )
plt.scatter(data[cluster_pred==2,0],data[cluster_pred==2,1], s = 100, c = 'green', label ='cluster 3' )
plt.scatter(data[cluster_pred==3,0],data[cluster_pred==3,1], s = 100, c = 'cyan', label = 'cluster 4')
plt.scatter(data[cluster_pred==4,0],data[cluster_pred==4,1], s = 100, c = 'cyan', label = 'cluster 5')
plt.scatter(cluster_center[:,0],cluster_center[:,1], s = 300, c = 'yellow', label = 'Centroids')
plt.show()