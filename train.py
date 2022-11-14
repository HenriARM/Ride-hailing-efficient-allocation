from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

f = "./robotex5.csv"
df = pd.read_csv(f)
df = df.sort_values(by=["start_time"])
# TODO: set time as index

distortions = []
inertias = []
K = range(5, 20)

# tmp
X = df[:1000][["start_lat", "start_lng"]]
X_test = df[1000:1100][["start_lat", "start_lng"]]

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    distortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    inertias.append(kmeans.inertia_)

    print(kmeans.labels_)
    print(kmeans.cluster_centers_)
    print(kmeans.predict(X_test))

# plot distortions
plt.clf()
plt.plot(K, distortions, linestyle='-', marker='o')
plt.xlabel('K')
plt.title('Distortion')
plt.savefig('distortion.png')

# plot inertia
plt.clf()
plt.plot(K, inertias, linestyle='-', marker='o')
plt.xlabel('K')
plt.title('Inertia')
plt.savefig('inertia.png')
