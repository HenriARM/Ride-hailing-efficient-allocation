import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

NUM_CLUSTERS = 5
time_col = 'start_time'


def preprocess(df):
    # TODO: tmp ignore noise orders (clustered with -1)
    df = df[df['cluster'] != -1.0]

    # group all rows based on (day, hour, cluster_idx) and sum ride value
    df['time'] = pd.to_datetime(df[time_col].dt.date) + pd.to_timedelta(df[time_col].dt.hour, unit='h')
    df = df[['time', 'cluster', 'ride_value']].groupby(['time', 'cluster']).sum().reset_index()
    # hour 0-23
    df['hour'] = df['time'].dt.hour
    # day of week 0-6
    df['dow'] = df['time'].dt.day_of_week

    # make OHE from hour, dow, cluster indexes
    # TODO: tmp dummys
    hour_ohe = pd.get_dummies(pd.Series(list(df['hour']) + [i for i in range(24)]), prefix='hour').iloc[:-24]
    dow_ohe = pd.get_dummies(pd.Series(list(df['dow']) + [i for i in range(7)]), prefix='dow').iloc[:-7]
    cluster_ohe = pd.get_dummies(df['cluster'], prefix='cluster')
    X = pd.concat([hour_ohe, dow_ohe, cluster_ohe], axis=1)
    y = df['ride_value']

    # X shape 24 (hours) + 7 (dows) + cluster num
    X = X.to_numpy()
    y = y.to_numpy().astype(np.float32)[..., np.newaxis]

    # normalize ride_value
    scaler = MinMaxScaler()
    y = scaler.fit_transform(y)
    print(f'min: {scaler.data_min_} max: {scaler.data_max_}')
    return X, y[:, 0]


def main():
    f = 'data/data_clustered.csv'
    df = pd.read_csv(f, parse_dates=[time_col])
    X, y = preprocess(df)

    # split train/test/eval to 70/20/10
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=1, test_size=0.1)
    # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
    model = MLPRegressor(
        random_state=1,
        learning_rate_init=0.001,
        activation='relu',
        solver='adam',
        verbose=True,
        shuffle=True,
        max_iter=500
    )
    model.fit(X_train, y_train)
    # save model
    pickle.dump(model, open('model.pickle', 'wb'))
    # TODO: add other metrics too except R2, e.x. MSE for X_test
    print(model.predict(X_test[:2]))
    # R2 score
    print(model.score(X_test, y_test))
    # TODO: train/test plots


def choose_riders_destination(lat, lon, time, model_path, centroid_path):
    # TODO: rename clusters_val, clusters_dist
    # load centroids from json file
    f = open('centroids.json')
    centroids = json.load(f)
    print(centroids)

    # calculate riders distance to all clusters
    from dbscan import haversine_distance
    dist_to_centroids = []
    for centroid in centroids:
        # TODO: check havers dist impl
        dist_to_centroids.append(haversine_distance(centroid['lon'], centroid['lat'], lon, lat))
    print(dist_to_centroids)
    # TODO: In what measure is output of havers dist? radians?

    # get day, hour from time
    hour = time.hour  # 0-23
    dow = time.weekday()  # Monday is 0 and Sunday is 6

    # convert to one-hot-encoding (add at the end all day and hour ranges so correct ohe shape would be generated)
    hour_ohe = pd.get_dummies(pd.Series([hour] + [i for i in range(24)]), prefix='hour').iloc[0]
    dow_ohe = pd.get_dummies(pd.Series([dow] + [i for i in range(7)]), prefix='dow').iloc[0]

    # load MLP
    model = pickle.load(open(model_path, 'rb'))

    # inference MLP (day, hour, centroids_idx) -> centroid sum ride value
    clusters_val = []
    for centroid_idx in range(len(centroids)):
        centroid_ohe = \
            pd.get_dummies(pd.Series([centroid_idx] + [i for i in range(len(centroids))]), prefix='centroid').iloc[0]
        X = np.concatenate((hour_ohe.to_numpy(), dow_ohe.to_numpy(), centroid_ohe.to_numpy()))
        X = X[np.newaxis, ...]
        clusters_val.append(model.predict(X)[0])
    print('cluster vals:', clusters_val)
    assert all(v > 0 for v in clusters_val)

    # chose best cluster min distance + max value
    # can also use min arctan(dist/val), normalize?
    clusters_val = np.asarray(clusters_val)
    dist_to_centroids = np.asarray(dist_to_centroids)
    score = np.sqrt(np.square(1 - clusters_val) + np.square(dist_to_centroids))
    best_cluster_idx = np.argmin(score)
    print(f'best cluster to move is n{best_cluster_idx} with centroid: {centroids[best_cluster_idx]}')


if __name__ == '__main__':
    # main()
    choose_riders_destination(
        lat=60.44011360274726,
        lon=23.72985704604993,
        time=datetime.now(),
        model_path='model.pickle',
        centroid_path='centroids.json'
    )

# TODO: PCA before training 24 + 7 + num cluster
