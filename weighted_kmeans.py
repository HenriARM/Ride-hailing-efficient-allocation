from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

f = './robotex5.csv'
df = pd.read_csv(f)
# df = df.sort_values(by=['start_time']).reset_index(drop=True)
df = df.dropna()

# check ride values are always positive number
assert (df['ride_value'] > 0).all()
# check lat in range [-90,90] and long in range [-180,180]
assert (df['start_lat'] >= -90).all() and (df['start_lat'] <= 90).all()
assert (df['start_lng'] >= -180).all() and (df['start_lng'] <= 180).all()

# remove outliers based on order price
# df['ride_value'].index[np.abs(stats.zscore(df['ride_value'])) > 3].tolist()
df = df[np.abs(stats.zscore(df['ride_value'])) < 3]

# TODO: set time as index

distortions = []
inertias = []
K = range(5, 20)

# tmp
LEN = 1000
X = df[:LEN][['start_lat', 'start_lng']]
X_test = df[LEN:LEN + 100][['start_lat', 'start_lng']]

weights = df[:LEN][['ride_value']]
weights = MinMaxScaler().fit_transform(weights)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X, sample_weight=weights.ravel())
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
