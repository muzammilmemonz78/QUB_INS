import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kmodes.kprototypes import KPrototypes

marketing_df = pd.read_csv('data assignment clean.csv')

#print(marketing_df.head(), marketing_df.tail())

marketing_df=marketing_df.drop(['CustomerID', "TravelType",'Occupation','GivenName','MiddleInitial','Surname'],axis=1)

print(marketing_df.head(), marketing_df.tail())

marketing_df = marketing_df.dropna()

print(marketing_df.info())

mark_array=marketing_df.values

mark_array[:,3] = mark_array[:,3].astype(float)
mark_array[:,6] = mark_array[:,6].astype(float)
mark_array[:,10] = mark_array[:,10].astype(float)
mark_array[:,11] = mark_array[:,11].astype(float)

print(mark_array)

kproto = KPrototypes(n_clusters=3, verbose=2, max_iter = 5)
clusters = kproto.fit_predict(mark_array, 
                              categorical=[0, 1, 2, 4, 5, 7, 8, 9, 12, 13])

print(kproto.cluster_centroids_)

cluster_dict=[]
for c in clusters:
    cluster_dict.append(c)
    
print(cluster_dict)

marketing_df['cluster']=cluster_dict

print(marketing_df[marketing_df['cluster']== 0].head(10), 
      marketing_df[marketing_df['cluster']== 0].tail(10))

print(marketing_df[marketing_df['cluster']== 1].head(10), 
      marketing_df[marketing_df['cluster']== 1].tail(10))

print(marketing_df[marketing_df['cluster']== 2].head(10), 
      marketing_df[marketing_df['cluster']== 2].tail(10))
