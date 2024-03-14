import pandas as pd
import numpy as np
import random
import sys
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler

#Clustering imports
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
#SKLEARN CLUSTERING TOOLS: https://scikit-learn.org/stable/modules/clustering.html

#Loading Data from calibrated_constrained_parameters.csv file
df = pd.read_csv (r'../../data/calibrated_constrained_parameters.csv')

#Only keep the columns relevant for parameters in 'theta'
df = df[['clim_gamma','clim_c1','clim_c2','clim_c3','clim_kappa1','clim_kappa2','clim_kappa3','clim_epsilon','clim_sigma_eta','clim_sigma_xi']]
labels = df.columns


scaler = StandardScaler()
model = scaler.fit(df)
data = pd.DataFrame(model.transform(df))

S=data.shape
nrows = S[0]
ncols = S[1]


#Clustering
PERCENT = [1]
LIST = [];
for percent in PERCENT:
  List = []
  #percent: How much of the dataset do we consider for clustering (useful if many rows)
  dfsample = data.sample(n=int(nrows*percent))

  NCLUSTERS = [3, 4, 5, 6] #Number of clusters equivalent to number of perspectives 
  for nclusters in NCLUSTERS:
      
    #Clustering
    print("Round ", nclusters, " with max round being 250 (percent =",percent,")")
    mixt = KMeans(n_clusters=nclusters, init='k-means++', random_state=0).fit(dfsample)
    
    #For each row, classify according to clustering A[ind_row]=ind_cluster
    A=mixt.predict(dfsample)
    
    #S is an array of arrays. S -> Cluster Index -> Copies of actual rows in the cluster (from corresponding dfsample)
    ind=0;
    S = [[] for i in range(np.max(A)+1)]
    for i in dfsample.iloc():
      c = A[ind]
      ind +=1
      S[c].append(i)

    # List counts the number of elements from the dataset in each cluster (from corresponding dfsample) 
    liste = []
    for s in S:
      n = len(s);
      liste.append(n)
    List.append(liste.copy())

  #Containt the number of elements in each cluster for each choice of NCLUSTER and each choice of PERCENT 
  # LIST -> ind_PERCENT -> ind_NCLUSTER -> ind_cluster -> number of elements in the cluster (from corresponding dfsample) 
  LIST += [List.copy()];

#Study of the size of clusters
SIZZ = [] #Constains for each PERCENT, the number of clusters of relevent size 
"""
The 'relevant size' is defined as 20% of the uniform repartition of row amongst all clusters
e.g. if I have 100 rows and try to create 4 clusters, if my dataset is balanced I expect 25 elements in each cluster, 
as dataset is probably imbalanced, I have to use a heuristic to assess the relevance of clusters, thus considering
that clusters are only relevent if of size at least 20% of the expected ideal size. That would mean, in this example, of
size at least 5 elements.
"""

indd = 0 #index for PERCENT

#Going through the PERCENT
for Llist in LIST:

  Sizz = [0 for i in NCLUSTERS];
  ind = 0 #index for NCLUSTERS
  #Going through the NCLUSTERS
  for L in Llist:
    #Going through the ind_cluster
    for l in L:
      if l>(nrows*PERCENT[indd]/NCLUSTERS[ind]*0.2): #TOTAL NUMBER OF FEATURES CONSIDERED / TOTAL NUMBER OF CLUSTERS * MINIMUM SIZE IN REGARD TO UNIFORM DISTRIBUTION OVER ALL CLUSTERS
        Sizz[ind]+=1                                  # Note that the dataset might not represent all classes at the same rate (imbalanced), so you want to keep the minimum size pretty low...
    ind += 1
  SIZZ += [Sizz.copy()]

  plt.plot(NCLUSTERS,Sizz,NCLUSTERS,NCLUSTERS)
  plt.legend(["clusters of size > 0.2*uniform repartition", "all sizes"])
  plt.title("Clusters from Kmeans considering %i %% of the dataset" % (100*PERCENT[indd]))
  plt.xlabel("Number of clusters")
  plt.ylabel("Number of significant clusters")
  plt.show()
  indd += 1
  
#Visualizing clusters 
#(Can comment following lines if you want to visualise results directly from the last clustering in the loops aboves)
percent = 1;
dfsample = data.sample(n=int(nrows*percent))
mixt = KMeans(n_clusters=3, init='k-means++', random_state=0).fit(dfsample)
A=mixt.fit_predict(dfsample)
ind=0;

dfsample_rescaled = df;
S = [[] for i in range(np.max(A)+1)]
for i in dfsample_rescaled.iloc():
  c = A[ind]
  ind +=1
  S[c].append(i)
#(Commenting for vizualisation of last clustering ends here)

# Let's look at cluster...
ind = 0;
fig, ax = plt.subplots(figsize=(7, 7))
STRG = []
for s in S:
  ind += 1
  if ind%1==0:
    mean_s = np.array(s).mean(axis=0)
    std_s = np.array(s).std(axis=0)
    
    
    print("cluster",ind," :", pd.DataFrame(mean_s, index=labels))

    # Number of variables we're plotting.
    num_vars = len(labels)-1

    # Split the circle into even parts and save the angles
    # so we know where to put each axis.
    angles = np.linspace(0, 2 * np.pi, num_vars+1, endpoint=False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    # and append the start value to the end.
    mean_s += mean_s[:1]
    angles += mean_s[:1]

    # Uncomment below to create separate plots for each clusters
    #fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_xticks([i for i in range(10)])
    ax.set_xticklabels(labels.tolist(), rotation=20)
    # Draw the outline of our data.
    ax.errorbar( x=[i for i in range(10)], y=mean_s, yerr=std_s, fmt='o')
    # Fill it in.
    #ax.fill(angles[:-1], mean_s, alpha=0.25)
    strg = ["cluster "+str(ind)]
    STRG += [strg]
    plt.title(strg)
    plt.legend(STRG)


mean_s = df.mean()
std_s = df.std()
ax.errorbar( x=[i for i in range(10)], y=mean_s, yerr=std_s, fmt='o')
strg = ["full dataset"]
STRG += [strg]
plt.title(strg)
plt.legend(STRG)
#Experimenting over the clusters with FaIR


fig1, axs = plt.subplots(len(labels),len(labels))


n = 0
for l1 in labels:
    m=0;
    for l2 in labels:
        for i in [0,1,2]:
            cluster_elements = df[A == i]
            
            axs[n,m].scatter(cluster_elements[l1] , cluster_elements[l2], label = i)
        m +=1
        
    n += 1
plt.show()