# TODO: convex hull around all the points using geopandas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

f = './robotex5.csv'
df = pd.read_csv(f)
df = df.sort_values(by=['start_time']).reset_index(drop=True)
df = df[:1000]
coords = df[['start_lat', 'start_lng']].to_numpy()

# https://stackoverflow.com/questions/12180290/convert-kilometers-to-radians
kms_per_radian = 6371.0088
epsilon = 1.5 / kms_per_radian
db = DBSCAN(eps=epsilon,
            min_samples=1,
            algorithm='ball_tree',
            metric='haversine').fit(np.radians(coords))
cluster_labels = db.labels_
num_clusters = len(set(cluster_labels))
clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
print('Number of clusters: {}'.format(num_clusters))

# from geopy.distance import great_circle
# from shapely.geometry import MultiPoint
#
# def get_centermost_point(cluster):
#     centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
#     centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
#     return tuple(centermost_point)
#
#
# centermost_points = clusters.map(get_centermost_point)
#
# lats, lons = zip(*centermost_points)
# rep_points = pd.DataFrame({'lon': lons, 'lat': lats})
# rs = rep_points.apply(lambda row: df[(df['lat'] == row['lat']) and (df['lon'] == row['lon'])].iloc[0], axis=1)
