# Clustering
## Machine Learning | Hands-on Clustering Analysis with Python - KMeans, DBSCAN, Hierarchical &amp; More.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster



file_path = '/content/EastWestAirlines.xlsx' # Replace with your file path
excel_df = pd.ExcelFile(file_path)
excel_data = pd.ExcelFile(file_path)
df= excel_data.parse('data')
df

df.head()

df.isnull().sum()

df.info()

df.describe()

df.size

df.shape

df.isnull().sum()

df.duplicated().sum()

def hist(df, columns):
  plt.figure(figsize=(30, 20))  # Corrected 'figzise' to 'figsize'
  df.hist(bins=15)
  plt.suptitle('Histogram of Numeric Features')
  plt.show()

hist(df,df.columns)

def bar(df,columns):
    plt.figure(figsize=(20,10))
    sns.barplot(data=df)
    plt.show()

bar(df,df.columns)

def scatter(df,columns):
    plt.figure(figsize=(20,10))
    sns.scatterplot(data=df)
    plt.show()

scatter(df,df.columns)

def pair(df,columns):
    plt.figure(figsize=(20,20))
    sns.pairplot(data=df)
    plt.suptitle("Scatter Plot")
    plt.show()

pair(df,df.columns)

def kde(df,columns):
    plt.figure(figsize=(10,5))
    sns.kdeplot(data=df)
    plt.show()

kde(df,df.columns)

def box_plot(df,columns):
    plt.figure(figsize=(20,10))
    sns.boxplot(data=df,color='grey')
    plt.show()

box_plot(df,df.columns)

std=StandardScaler()
d=pd.DataFrame(std.fit_transform(df),columns=df.columns)
d

box_plot(d,d.columns)


import scipy.stats as stats
def o_r(df,columns):
    thres=2
    z_score=stats.zscore(df[columns])
    return df[(abs(z_score)<thres).all(axis=1)]
    df=o_r(df,df.columns)
df

d

d=o_r(d,d.columns)

d

box_plot(d,d.columns)

def corr (df,columns):
    plt.figure(figsize=(20,10))
    sns.heatmap(df.corr(),annot=True)
    plt.show()

corr(df,df.columns)

corr(d,d.columns)

features=d.drop(columns=['Flight_miles_12mo','ID#','Balance'])

kmeans=KMeans(n_clusters=5,random_state=40)
cluster=kmeans.fit(features)

cluster

np.unique(cluster.labels_)

features

silhouette_score(features,cluster.labels_)

## Optimal Value For K

n_clusters=range(1,11)
inertia=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(features)
    inertia.append(kmeans.inertia_)
print(inertia)

plt.plot(inertia,n_clusters)
plt.xlabel("No of features")
plt.ylabel("clusters")
plt.title("optimal value for K")
plt.show()

import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(features,method='ward'))


from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
agg=AgglomerativeClustering(n_clusters=5,linkage='ward')
cluster_hc=agg.fit(features)

cluster_hc

cluster_hc.labels_

silhouette_score(features,cluster_hc.labels_)

features

output=[]
eps= np.arange(1.0,2.0,0.1)
min_node= range(1,10)
for ep in eps:
    for mn in min_node:
        labels= DBSCAN(eps=ep, min_samples=mn).fit(features)
        set=labels.labels_
        score= silhouette_score(features,set)
        output.append([ep,mn,score])
print(output)

a=pd.DataFrame(output)

a

sil_score=a.iloc[:,2].max()
a[a[2]==sil_score]

labels=DBSCAN(eps=1.9,min_samples=3)

cluster_db=labels.fit(features)

np.unique(cluster_db.labels_)

silhouette_score(features,cluster_db.labels_)

import warnings
warnings.filterwarnings('ignore')


df = df[df.index.isin(features.index)]
# Now you can assign the cluster labels
df['KMeans_Cluster'] = cluster.labels_
df['Hierarchical_Cluster'] = cluster_hc.labels_
df['DBSCAN_Cluster'] = cluster_db.labels_

print(df.groupby('KMeans_Cluster').mean())
print(df.groupby('Hierarchical_Cluster').mean())
print(df.groupby('DBSCAN_Cluster').mean())

df

plt.scatter(df.iloc[:, 1], df.iloc[:, 6], c=df['KMeans_Cluster'], cmap='viridis')
plt.title('K-Means Clustering')
plt.show()

plt.scatter(df.iloc[:, 1], df.iloc[:, 6], c=df['Hierarchical_Cluster'], cmap='viridis')
plt.title('Hierachical Clustering')
plt.show()

plt.scatter(df.iloc[:, 1], df.iloc[:, 6], c=df['DBSCAN_Cluster'], cmap='viridis')
plt.title('DBSCAN Clustering')
plt.show()

silhouette_score(features,cluster.labels_)

silhouette_score(features,cluster_hc.labels_)

silhouette_score(features,cluster_db.labels_)

## Higher silhouette scores indicate better-defined clusters.
### DBSCAN may give lower scores if the data is noisy or contains many outliers.
## K-Means generally performs well on spherical clusters but may struggle with irregular shapes.
### Hierarchical clustering works well with smaller datasets but may become slow with large data.
