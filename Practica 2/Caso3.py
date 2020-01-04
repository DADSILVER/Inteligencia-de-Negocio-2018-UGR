# -*- coding: utf-8 -*-
"""
Autor:
    David Castro Salazar
Contenido:
    Caso de estudio 1 realizado para
    la practica 2 de la asignatura 
    de Inteligencia de Negocio
"""

'''
Documentación sobre clustering en Python:
    http://scikit-learn.org/stable/modules/clustering.html
    http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/
    http://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
    https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/
'''

import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import estimate_bandwidth
from sklearn import metrics
from scipy.cluster.hierarchy import dendrogram, linkage
from math import floor
import seaborn as sns


#ALGORITMO 1



def norm_to_zero_one(df):
    return (df - df.min()) * 1.0 / (df.max() - df.min())

censo = pd.read_csv('censo_granada.csv')
censo = censo.replace(np.NaN,0) #los valores en blanco realmente son otra categoría que nombramos como 0

#seleccionar casos

subset = censo.loc[censo['INTERNET']==2]
subset = subset.loc[subset['EDAD']>18]
subset = subset.loc[subset['ESTADO']>3]
subset = subset.loc[subset['RELA']<6]
subset = subset.loc[subset['TELEF']==1]

#seleccionar variables de interés para clustering
usadas = ['EDAD', 'NOCU','NPARAIN','H0515', 'H1624','SUT']
X = subset[usadas]

X_normal = X.apply(norm_to_zero_one)






#ALGORITMO 1.1

print('----- Ejecutando k-Means 3',end='')
k_means = KMeans(init='k-means++', n_clusters=3, n_init=5)



t = time.time()
cluster_predict = k_means.fit_predict(X_normal) 
tiempo = time.time() - t
print(": {:.2f} segundos, ".format(tiempo), end='')
metric_CH = metrics.calinski_harabaz_score(X_normal, cluster_predict)
print("Calinski-Harabaz Index: {:.3f}, ".format(metric_CH), end='')
#el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, se puede seleccionar una muestra, p.ej., el 10%
metric_SC = metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', sample_size=floor(0.2*len(X)), random_state=123456)
print("Silhouette Coefficient: {:.5f}".format(metric_SC))

#se convierte la asignación de clusters a DataFrame
clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])

print("Tamaño de cada cluster:")
size=clusters['cluster'].value_counts()
for num,i in size.iteritems():
   print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))

centers = pd.DataFrame(k_means.cluster_centers_,columns=list(X))
centers_desnormal = centers.copy()

#se convierten los centros a los rangos originales antes de normalizar
for var in list(centers):
    centers_desnormal[var] = X[var].min() + centers[var] * (X[var].max() - X[var].min())

sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')
plt.savefig("kmeans_heatmap3.1.png")
plt.clf()

print("---------- Preparando el scatter matrix...")
#se añade la asignación de clusters como columna a X
X_kmeans = pd.concat([X, clusters], axis=1)
sns.set()
variables = list(X_kmeans)
variables.remove('cluster')
sns_plot = sns.pairplot(X_kmeans, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
plt.savefig("kmeans3.1.png")
plt.clf()


print("")




#ALGORITMO 1.2

print('----- Ejecutando k-Means 5',end='')
k_means = KMeans(init='k-means++', n_clusters=5, n_init=5)



t = time.time()
cluster_predict = k_means.fit_predict(X_normal) 
tiempo = time.time() - t
print(": {:.2f} segundos, ".format(tiempo), end='')
metric_CH = metrics.calinski_harabaz_score(X_normal, cluster_predict)
print("Calinski-Harabaz Index: {:.3f}, ".format(metric_CH), end='')
#el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, se puede seleccionar una muestra, p.ej., el 10%
metric_SC = metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', sample_size=floor(0.2*len(X)), random_state=123456)
print("Silhouette Coefficient: {:.5f}".format(metric_SC))

#se convierte la asignación de clusters a DataFrame
clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])

print("Tamaño de cada cluster:")
size=clusters['cluster'].value_counts()
for num,i in size.iteritems():
   print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))

centers = pd.DataFrame(k_means.cluster_centers_,columns=list(X))
centers_desnormal = centers.copy()

#se convierten los centros a los rangos originales antes de normalizar
for var in list(centers):
    centers_desnormal[var] = X[var].min() + centers[var] * (X[var].max() - X[var].min())

sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')
plt.savefig("kmeans_heatmap3.2.png")
plt.clf()

print("---------- Preparando el scatter matrix...")
#se añade la asignación de clusters como columna a X
X_kmeans = pd.concat([X, clusters], axis=1)
sns.set()
variables = list(X_kmeans)
variables.remove('cluster')
sns_plot = sns.pairplot(X_kmeans, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
plt.savefig("kmeans3.2.png")
plt.clf()
print("")





#ALGORITMO 1.3

print('----- Ejecutando k-Means 8',end='')
k_means = KMeans(init='k-means++', n_clusters=8, n_init=5)



t = time.time()
cluster_predict = k_means.fit_predict(X_normal) 
tiempo = time.time() - t
print(": {:.2f} segundos, ".format(tiempo), end='')
metric_CH = metrics.calinski_harabaz_score(X_normal, cluster_predict)
print("Calinski-Harabaz Index: {:.3f}, ".format(metric_CH), end='')
#el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, se puede seleccionar una muestra, p.ej., el 10%
metric_SC = metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', sample_size=floor(0.2*len(X)), random_state=123456)
print("Silhouette Coefficient: {:.5f}".format(metric_SC))

#se convierte la asignación de clusters a DataFrame
clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])

print("Tamaño de cada cluster:")
size=clusters['cluster'].value_counts()
for num,i in size.iteritems():
   print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))

centers = pd.DataFrame(k_means.cluster_centers_,columns=list(X))
centers_desnormal = centers.copy()

#se convierten los centros a los rangos originales antes de normalizar
for var in list(centers):
    centers_desnormal[var] = X[var].min() + centers[var] * (X[var].max() - X[var].min())

sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')
plt.savefig("kmeans_heatmap3.3.png")
plt.clf()

print("---------- Preparando el scatter matrix...")
#se añade la asignación de clusters como columna a X
X_kmeans = pd.concat([X, clusters], axis=1)
sns.set()
variables = list(X_kmeans)
variables.remove('cluster')
sns_plot = sns.pairplot(X_kmeans, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
plt.savefig("kmeans3.3.png")
plt.clf()
print("")



#ALGORITMO 2



print('----- Ejecutando Birch',end='')
brc = Birch(threshold=0.2, branching_factor=50, n_clusters=5, compute_labels=True, copy=True)
t = time.time()
cluster_predict = brc.fit_predict(X_normal) 
tiempo = time.time() - t
print(": {:.2f} segundos, ".format(tiempo), end='')
metric_CH = metrics.calinski_harabaz_score(X_normal, cluster_predict)
print("Calinski-Harabaz Index: {:.3f}, ".format(metric_CH), end='')
#el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, se puede seleccionar una muestra, p.ej., el 10%
metric_SC = metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', sample_size=floor(0.2*len(X)), random_state=123456)
print("Silhouette Coefficient: {:.5f}".format(metric_SC))

#se convierte la asignación de clusters a DataFrame
clusters = pd.DataFrame(cluster_predict,index=X_normal.index,columns=['cluster'])

print("Tamaño de cada cluster:")
size=clusters['cluster'].value_counts()
for num,i in size.iteritems():
   print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))
   

X_brc = pd.concat([X_normal, clusters], axis=1)


labels = brc.labels_  

X_brc_normal = pd.concat([X_normal, clusters], axis=1)
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

centros=[]

for r in range(0,n_clusters_):
    centros.append(np.mean(X_brc_normal.loc[X_brc_normal['cluster']==r]))
    

centers = pd.DataFrame(centros ,columns=list(X_normal))
centers_desnormal = centers.copy()

#se convierten los centros a los rangos originales antes de normalizar
for var in list(centers):
    centers_desnormal[var] = X[var].min() + centers[var] * (X[var].max() - X[var].min())

sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')
plt.savefig("brc_heatmap3.png")
plt.clf()

print("---------- Preparando el scatter matrix...")
#se añade la asignación de clusters como columna a X

sns.set()
variables = list(X_brc)
variables.remove('cluster')
sns_plot = sns.pairplot(X_brc, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
plt.savefig("brc3.png")
plt.clf()



Z = linkage(X_kmeans, 'ward')
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=12,  # show only the last p merged clusters
    show_leaf_counts=False,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.savefig("brc_dendrograma3.png")
plt.clf()



print("")





#ALGORITMO 3.1



#band=estimate_bandwidth(X_normal, n_samples=3000)

print('----- Ejecutando MeanShift bandwidth=0.45',end='')
MSC = MeanShift(bandwidth=0.45, bin_seeding=True)



t = time.time()
cluster_predict = MSC.fit_predict(X_normal) 
tiempo = time.time() - t
print(": {:.2f} segundos, ".format(tiempo), end='')
metric_CH = metrics.calinski_harabaz_score(X_normal, cluster_predict)
print("Calinski-Harabaz Index: {:.3f}, ".format(metric_CH), end='')
#el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, se puede seleccionar una muestra, p.ej., el 10%
metric_SC = metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', sample_size=floor(0.2*len(X)), random_state=123456)
print("Silhouette Coefficient: {:.5f}".format(metric_SC))

#se convierte la asignación de clusters a DataFrame
clusters = pd.DataFrame(cluster_predict,index=X_normal.index,columns=['cluster'])

print("Tamaño de cada cluster:")
size=clusters['cluster'].value_counts()
for num,i in size.iteritems():
   print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))
   
   
   
labels = MSC.labels_  

X_MSC_normal = pd.concat([X_normal, clusters], axis=1)
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_clusters_-=1

centros=[]

for r in range(0,n_clusters_):
    centros.append(np.mean(X_MSC_normal.loc[X_MSC_normal['cluster']==r]))


centers = pd.DataFrame(MSC.cluster_centers_,columns=list(X))
centers_desnormal = centers.copy()

#se convierten los centros a los rangos originales antes de normalizar
for var in list(centers):
   centers_desnormal[var] = X[var].min() + centers[var] * (X[var].max() - X[var].min())

sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')
plt.savefig("MSC_heatmap3.1.png")
plt.clf()

print("---------- Preparando el scatter matrix...")
#se añade la asignación de clusters como columna a X
X_MSC = pd.concat([X_normal, clusters], axis=1)
sns.set()
variables = list(X_MSC)
variables.remove('cluster')
sns_plot = sns.pairplot(X_MSC, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
plt.savefig("MSC3.1.png")
plt.clf()
print("")





#ALGORITMO 3.2



band=estimate_bandwidth(X_normal, n_samples=300)

print('----- Ejecutando MeanShift con estimate_bandwidth(X_normal, n_samples=300)',end='')
MSC = MeanShift(bandwidth=band, bin_seeding=True)



t = time.time()
cluster_predict = MSC.fit_predict(X_normal) 
tiempo = time.time() - t
print(": {:.2f} segundos, ".format(tiempo), end='')
metric_CH = metrics.calinski_harabaz_score(X_normal, cluster_predict)
print("Calinski-Harabaz Index: {:.3f}, ".format(metric_CH), end='')
#el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, se puede seleccionar una muestra, p.ej., el 10%
metric_SC = metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', sample_size=floor(0.2*len(X)), random_state=123456)
print("Silhouette Coefficient: {:.5f}".format(metric_SC))

#se convierte la asignación de clusters a DataFrame
clusters = pd.DataFrame(cluster_predict,index=X_normal.index,columns=['cluster'])

print("Tamaño de cada cluster:")
size=clusters['cluster'].value_counts()
for num,i in size.iteritems():
   print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))
   
   
   
labels = MSC.labels_  

X_MSC_normal = pd.concat([X_normal, clusters], axis=1)
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_clusters_-=1

centros=[]

for r in range(0,n_clusters_):
    centros.append(np.mean(X_MSC_normal.loc[X_MSC_normal['cluster']==r]))


centers = pd.DataFrame(MSC.cluster_centers_,columns=list(X))
centers_desnormal = centers.copy()

#se convierten los centros a los rangos originales antes de normalizar
for var in list(centers):
   centers_desnormal[var] = X[var].min() + centers[var] * (X[var].max() - X[var].min())

sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')
plt.savefig("MSC_heatmap3.2.png")
plt.clf()

print("---------- Preparando el scatter matrix...")
#se añade la asignación de clusters como columna a X
X_MSC = pd.concat([X_normal, clusters], axis=1)
sns.set()
variables = list(X_MSC)
variables.remove('cluster')
sns_plot = sns.pairplot(X_MSC, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
plt.savefig("MSC3.2.png")
plt.clf()
print("")






#ALGORITMO 3.3


band=estimate_bandwidth(X_normal, n_samples=10000)

print('----- Ejecutando MeanShift con estimate_bandwidth(X_normal, n_samples=10000)',end='')
MSC = MeanShift(bandwidth=band, bin_seeding=True)



t = time.time()
cluster_predict = MSC.fit_predict(X_normal) 
tiempo = time.time() - t
print(": {:.2f} segundos, ".format(tiempo), end='')
metric_CH = metrics.calinski_harabaz_score(X_normal, cluster_predict)
print("Calinski-Harabaz Index: {:.3f}, ".format(metric_CH), end='')
#el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, se puede seleccionar una muestra, p.ej., el 10%
metric_SC = metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', sample_size=floor(0.2*len(X)), random_state=123456)
print("Silhouette Coefficient: {:.5f}".format(metric_SC))

#se convierte la asignación de clusters a DataFrame
clusters = pd.DataFrame(cluster_predict,index=X_normal.index,columns=['cluster'])

print("Tamaño de cada cluster:")
size=clusters['cluster'].value_counts()
for num,i in size.iteritems():
   print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))
   
   
   
labels = MSC.labels_  

X_MSC_normal = pd.concat([X_normal, clusters], axis=1)
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_clusters_-=1

centros=[]

for r in range(0,n_clusters_):
    centros.append(np.mean(X_MSC_normal.loc[X_MSC_normal['cluster']==r]))


centers = pd.DataFrame(MSC.cluster_centers_,columns=list(X))
centers_desnormal = centers.copy()

#se convierten los centros a los rangos originales antes de normalizar
for var in list(centers):
   centers_desnormal[var] = X[var].min() + centers[var] * (X[var].max() - X[var].min())

sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')
plt.savefig("MSC_heatmap3.3.png")
plt.clf()

print("---------- Preparando el scatter matrix...")
#se añade la asignación de clusters como columna a X
X_MSC = pd.concat([X_normal, clusters], axis=1)
sns.set()
variables = list(X_MSC)
variables.remove('cluster')
sns_plot = sns.pairplot(X_MSC, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
plt.savefig("MSC3.3.png")
plt.clf()
print("")





#ALGORITMO 4




print('----- Ejecutando AgglomerativeClustering',end='')
Agclus = AgglomerativeClustering(n_clusters=5, affinity='euclidean', memory=None,
                              connectivity=None, compute_full_tree='auto', 
                              linkage='ward', pooling_func='deprecated')



t = time.time()
cluster_predict = Agclus.fit_predict(X_normal) 
tiempo = time.time() - t
print(": {:.2f} segundos, ".format(tiempo), end='')
metric_CH = metrics.calinski_harabaz_score(X_normal, cluster_predict)
print("Calinski-Harabaz Index: {:.3f}, ".format(metric_CH), end='')
#el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, se puede seleccionar una muestra, p.ej., el 10%
metric_SC = metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', sample_size=floor(0.2*len(X)), random_state=123456)
print("Silhouette Coefficient: {:.5f}".format(metric_SC))

#se convierte la asignación de clusters a DataFrame
clusters = pd.DataFrame(cluster_predict,index=X_normal.index,columns=['cluster'])

print("Tamaño de cada cluster:")
size=clusters['cluster'].value_counts()

for num,i in size.iteritems():
   print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))


   

X_Agclus_normal = pd.concat([X_normal, clusters], axis=1)

centros=[]

for r in range(0,5):
    centros.append(np.mean(X_Agclus_normal.loc[X_Agclus_normal['cluster']==r]))

centers = pd.DataFrame(centros,columns=list(X))
centers_desnormal = centers.copy()

#se convierten los centros a los rangos originales antes de normalizar
for var in list(centers):
   centers_desnormal[var] = X[var].min() + centers[var] * (X[var].max() - X[var].min())

sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')
plt.savefig("Agclus_heatmap3.png")
plt.clf()




print("---------- Preparando el scatter matrix...")
#se añade la asignación de clusters como columna a X
X_Agclus = pd.concat([X_normal, clusters], axis=1)
sns.set()
variables = list(X_Agclus)
variables.remove('cluster')
sns_plot = sns.pairplot(X_Agclus, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
plt.savefig("Agclus3.png")
plt.clf()


Z = linkage(X_kmeans, 'ward')
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=12,  # show only the last p merged clusters
    show_leaf_counts=False,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.savefig("Agclus_dendrograma3.png")
plt.clf()




print("")






#ALGORITMO 5





print('----- Ejecutando SpectralClustering',end='')
SpectClus = SpectralClustering(n_clusters=5, eigen_solver=None, random_state=None, 
                         n_init=20, gamma=1.0, affinity='rbf', n_neighbors=10, 
                         eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, 
                         kernel_params=None, n_jobs=None)

X_sample = X.sample(2000, random_state=123456);
X_sample_normal = X_sample.apply(norm_to_zero_one)



t = time.time()
cluster_predict = SpectClus.fit_predict(X_sample_normal) 
tiempo = time.time() - t
print(": {:.2f} segundos, ".format(tiempo), end='')
metric_CH = metrics.calinski_harabaz_score(X_sample_normal, cluster_predict)
print("Calinski-Harabaz Index: {:.3f}, ".format(metric_CH), end='')
#el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, se puede seleccionar una muestra, p.ej., el 10%
metric_SC = metrics.silhouette_score(X_sample_normal, cluster_predict, metric='euclidean', sample_size=floor(0.2*len(X)), random_state=123456)
print("Silhouette Coefficient: {:.5f}".format(metric_SC))

#se convierte la asignación de clusters a DataFrame
clusters = pd.DataFrame(cluster_predict,index=X_sample.index,columns=['cluster'])

print("Tamaño de cada cluster:")
size=clusters['cluster'].value_counts()
for num,i in size.iteritems():
   print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))



X_SpectClus_normal = pd.concat([X_normal, clusters], axis=1)

centros=[]

for r in range(0,5):
    centros.append(np.mean(X_SpectClus_normal.loc[X_SpectClus_normal['cluster']==r]))

centers = pd.DataFrame(centros,columns=list(X_sample))
centers_desnormal = centers.copy()

#se convierten los centros a los rangos originales antes de normalizar
for var in list(centers):
    centers_desnormal[var] = X_sample[var].min() + centers[var] * (X_sample[var].max() - X_sample[var].min())

sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')
plt.savefig("SpectClus_heatmap3.png")
plt.clf()






print("---------- Preparando el scatter matrix...")
#se añade la asignación de clusters como columna a X
X_SpectClus = pd.concat([X_sample, clusters], axis=1)
sns.set()
variables = list(X_SpectClus)
variables.remove('cluster')
sns_plot = sns.pairplot(X_SpectClus, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
plt.savefig("SpectClus3.png")
plt.clf()


print("")






