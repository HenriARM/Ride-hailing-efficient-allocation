import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy import stats

from geopy.distance import great_circle
# https://shapely.readthedocs.io/en/latest/reference/shapely.MultiPoint.html
from shapely.geometry import MultiPoint
from PIL import Image
import io
import folium
import random


def get_centermost_point(cluster):
    if len(cluster) == 0:
        return 0.0, 0.0
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)


def preprocess(df):
    df = df.dropna()
    # check ride values are always positive number
    assert (df['ride_value'] > 0).all()
    # check lat in range [-90,90] and long in range [-180,180]
    assert (df['start_lat'] >= -90).all() and (df['start_lat'] <= 90).all()
    assert (df['start_lng'] >= -180).all() and (df['start_lng'] <= 180).all()
    # remove outliers based on order price
    df = df[np.abs(stats.zscore(df['ride_value'])) < 3]
    df = df[['start_lat', 'start_lng']]
    # remove repretetive start locations
    # df = df.drop_duplicates()
    return df.to_numpy()


def main():
    f = 'data/robotex5.csv'
    time_col = 'start_time'
    df = pd.read_csv(f, parse_dates=[time_col])
    df = df.sort_values(by=['start_time']).reset_index(drop=True)
    df = df[:1000]
    coords = preprocess(df)

    # https://stackoverflow.com/questions/12180290/convert-kilometers-to-radians
    kms_per_radian = 6371.0088
    # epsilon = 1.5 / kms_per_radian
    epsilon = 0.08 / kms_per_radian
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN
    db = DBSCAN(eps=epsilon,
                min_samples=5,
                algorithm='ball_tree',
                metric='haversine').fit(np.radians(coords))
    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels)) - 1  # noisy samples are labeled with -1
    print('Number of clusters: {}'.format(num_clusters))
    # save clusters (key: cluster_idx, value: list of points)
    clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
    df['cluster'] = pd.Series(cluster_labels)
    df.to_csv('data/data_clustered.csv', index=False)

    # plot all clusters with different colors
    r = lambda: random.randint(0, 255)
    # colors = ['red', 'green', 'blue', 'yellow', 'orange']
    TALLINN_LAT_LONG = [59.436962, 24.753574]
    m = folium.Map(location=TALLINN_LAT_LONG, tiles='openstreetmap', zoom_start=13)
    for idx in range(len(clusters)):
        print(f'Cluster {idx}')
        color = '#%02X%02X%02X' % (r(), r(), r())
        print(color)
        for point in clusters[idx]:
            folium.Circle(
                location=list(point),
                radius=1,
                color=color  # colors[idx]
            ).add_to(m)
    # plot centroids too
    # https://en.wikipedia.org/wiki/Great-circle_distance
    centermost_points = clusters.map(get_centermost_point)
    for centroid in list(centermost_points):
        folium.Circle(
            location=list(centroid),
            radius=10,
            color='red'
        ).add_to(m)
    # save
    image_data = m._to_png(5)
    image = Image.open(io.BytesIO(image_data))
    image.save(f'./map.png')


if __name__ == '__main__':
    main()
