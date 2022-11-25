import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

NUM_CLUSTERS = 5
time_col = 'start_time'


def preprocess(df):
    # group all rows based on (day, hour, cluster_idx) and sum ride value
    df['time'] = pd.to_datetime(df[time_col].dt.date) + pd.to_timedelta(df[time_col].dt.hour, unit='h')
    df = df[['time', 'cluster', 'ride_value']].groupby(['time', 'cluster']).sum().reset_index()
    df['hour'] = df['time'].dt.hour
    df['dow'] = df['time'].dt.day_of_week

    # make OHE from hour, dow, cluster indexes
    X = pd.get_dummies(df['hour'], prefix='hour')
    X = pd.concat([X, pd.get_dummies(df['dow'], prefix='dow')], axis=1)
    X = pd.concat([X, pd.get_dummies(df['cluster'], prefix='cluster')], axis=1)
    y = df['ride_value']
    return X.to_numpy(), y.to_numpy().astype(np.float32)


def main():
    f = 'data_clustered.csv'
    df = pd.read_csv(f, parse_dates=[time_col])
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)
    # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
    regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
    print(regr.predict(X_test[:2]))
    print(regr.score(X_test, y_test))
    # TODO: normalization


if __name__ == '__main__':
    main()
