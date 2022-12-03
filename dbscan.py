import json
import pickle
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
from sklearn.metrics import silhouette_score


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
    return df


def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    All args must be of equal length.
    Sourced from https://stackoverflow.com/a/29546836/11637704
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def save_centroids_as_json(points):
    json_dict = []
    for lat, lon in list(points):
        json_dict.append({'lat': lat, 'lon': lon})
    out_file = open('centroids.json', 'w')
    json.dump(json_dict, out_file, indent=4)


def random_color():
    r = lambda: random.randint(0, 255)
    color = '#%02X%02X%02X' % (r(), r(), r())
    return color


def save_model(model, model_filename):
    pickle.dump(model, open(model_filename, 'wb'))
    return model


def load_model(model_filename):
    return pickle.load(open(model_filename, 'rb'))


def create_map(data, cluster_col, num_clusters):
    m = folium.Map(location=[data['start_lat'].mean(), data['start_lng'].mean()], zoom_start=13, tiles='openstreetmap')
    cluster_colours = {-1: 'black'}
    for cluster_idx in range(num_clusters):
        cluster_colours[cluster_idx] = random_color()
    for _, row in data.iterrows():
        folium.CircleMarker(
            location=[row['start_lat'], row['start_lng']],
            radius=1,
            popup=row[cluster_col],
            color=cluster_colours[row[cluster_col]],
            fill=True,
            fill_color=cluster_colours[row[cluster_col]]
        ).add_to(m)
        # folium.Circle(
        #     location=[row['start_lat'], row['start_lng']],
        #     radius=1,
        #     color=cluster_colours[row[cluster_col]]
        # ).add_to(m)
    # # plot centroids too
    # for centroid in list(centermost_points):
    #     folium.Circle(
    #         location=list(centroid),
    #         radius=10,
    #         color='red'
    #     ).add_to(m)
    return m


def save_map(map):
    image_data = map._to_png(5)
    image = Image.open(io.BytesIO(image_data))
    image.save(f'map.png')


def cluster_eval(df, prediction_col):
    cluster_pred = df[prediction_col]
    print(f'Number of clusters found: {len(np.unique(cluster_pred))}')
    print(f'Number of outliers found: {len(cluster_pred[cluster_pred == -1])}')
    silhouette_wo_outliers = silhouette_score(df[cluster_pred != -1][["start_lat", "start_lng"]],
                                              cluster_pred[cluster_pred != -1])
    print(f'Silhouette Score ignoring outliers: {silhouette_wo_outliers}')
    print(f'Silhouette Score with outliers: {silhouette_score(df[["start_lat", "start_lng"]], cluster_pred)}')
    no_outliers = np.array([(counter + 2) * x if x == -1 else x for counter, x in enumerate(cluster_pred)])
    print(f'Silhouette Score outliers as singletons: {silhouette_score(df[["start_lat", "start_lng"]], no_outliers)}')


# def classify_outliers():
#     # apply a K nearest neighbor classifier on the outliers
#     # and assign them to the clusters which belong to their k nearest neighbor
#     # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
#     classifier = KNeighborsClassifier(n_neighbors=3)
#     data_train = data[data.CLUSTER_hdbscan != -1]
#     data_predict = data[data.CLUSTER_hdbscan == -1]
#     X_train = np.array(data_train[['LON', 'LAT']], dtype='float64')
#     y_train = np.array(data_train['CLUSTER_hdbscan'])
#
#     X_predict = np.array(data_predict[['LON', "LAT"]], dtype='float64')
#     classifier.fit(X_train, y_train)
#     pred = classifier.predict(X_predict)
#     data['CLUSTER_hybrid'] = data['CLUSTER_hdbscan']
#     data.loc[data.CLUSTER_hdbscan == -1, 'CLUSTER_hybrid'] = pred
#     data.head()


def plot_cluster_hist(df, cluster_col):
    df[cluster_col].value_counts().plot.hist(bins=70, alpha=0.5, label='hybrid')
    plt.legend()
    plt.grid(True)
    plt.title('Comparing Hybrid and K-Means Approaches')
    plt.xlabel('cluster sizes')
    plt.show()
    plt.savefig(f'{cluster_col}_hist.png')


# TODO: can be added as a metric on how dense cluster is (before adding outliers to it)
def cluster_avg_dist(latitudes, longitudes):
    """
    Function to calculate the average distance from the vertices of a convex hull] to the centroid of
    said convex hull.
    """
    centre_long = longitudes.mean()
    centre_lats = latitudes.mean()
    # collapse two points into line
    if len(latitudes) < 3:
        distances = haversine_distance(
            longitudes,
            latitudes,
            centre_long,
            centre_lats).mean()
    else:
        # convex hull
        convex_hull = ConvexHull([x for x in zip(latitudes, longitudes)])
        # now get co-ordinates of vertices
        vertex_longs = longitudes.iloc[convex_hull.vertices]
        vertex_lats = latitudes.iloc[convex_hull.vertices]
        # now get
        distances = haversine_distance(
            vertex_longs,
            vertex_lats,
            centre_long,
            centre_lats).mean()
    # return average distance
    return distances.mean()


def cluster_dbscan(coords):
    # https://stackoverflow.com/questions/12180290/convert-kilometers-to-radians
    kms_per_radian = 6371.0088
    # epsilon = 1.5 / kms_per_radian
    epsilon = 0.05 / kms_per_radian
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN
    model = DBSCAN(eps=epsilon,
                   min_samples=5,
                   algorithm='ball_tree',
                   metric='haversine').fit(np.radians(coords))
    cluster_pred = model.labels_
    return model, cluster_pred


# TODO:
def cluster_spectral():
    pass


# TODO:
def cluster_hdbscan():
    # # Hierarchical DBSCAN uses different set of hyperparameters to find different levels of density reducing outliers
    # import hdbscan
    #
    # model = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2, cluster_selection_epsilon=0.01)
    # class_predictions = model.fit_predict(X)
    # data['CLUSTER_hdbscan'] = class_predictions
    # data.head()
    pass


def main():
    f = 'data/robotex5.csv'
    time_col = 'start_time'
    df = pd.read_csv(f, parse_dates=[time_col])
    df = df.sort_values(by=['start_time']).reset_index(drop=True)
    df = df[:10000]
    # TODO: remove outliers before saving as clustered.csv + remove duplicates before clustering
    coords_df = preprocess(df)
    coords = coords_df.to_numpy()

    model, cluster_pred = cluster_dbscan(coords)
    save_model(model, 'cluster.pickle')

    num_clusters = len(np.unique(cluster_pred))
    df['cluster'] = pd.Series(cluster_pred)
    df['cluster'] = df['cluster'].astype('int64', errors='ignore')

    # TODO: tmp outliers
    df = df[:4042]
    df.to_csv('data/data_clustered.csv', index=False)

    cluster_eval(df, 'cluster')

    # https://en.wikipedia.org/wiki/Great-circle_distance
    # clusters (key: cluster_idx, value: list of points)
    clusters = pd.Series([coords[cluster_pred ==
                                 n] for n in range(num_clusters)])
    centermost_points = clusters.map(get_centermost_point)
    save_centroids_as_json(centermost_points)

    # Ideally we’d like small and dense clusters
    plot_cluster_hist(df, 'cluster')
    m = create_map(df, 'cluster', num_clusters)
    save_map(m)


if __name__ == '__main__':
    main()

# TODO: plot cluster not with points but whole area (maybe as a convex hull)