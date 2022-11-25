import pandas as pd
import geopandas as gpd
from PIL import Image
import io
import imageio
import glob

import folium
from folium.features import DivIcon

# https://www.latlong.net/place/tallinn-estonia-5882.html
TALLINN_LAT_LONG = [59.436962, 24.753574]
CIRCLE_RADIUS = 20
m = folium.Map(location=TALLINN_LAT_LONG, tiles='openstreetmap', zoom_start=13)

f = './robotex5.csv'
time_col = 'start_time'
df = pd.read_csv(f, parse_dates=[time_col])
df = df.sort_values(by=[time_col])

df = df[:2000]
df['hour'] = df[time_col].dt.hour
df['date'] = df[time_col].dt.date
df['time'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['start_lat'], df['start_lng']))
gdf = gdf[['time', 'geometry', 'ride_value']]
print(gdf.info())
print(gdf.head())

list_of_points = gdf.groupby(['time'])['geometry'].apply(list)
list_of_vals = gdf.groupby(['time'])['ride_value'].apply(list)
hours = list_of_points.reset_index()['time'].dt.hour
days_of_week = list_of_points.reset_index()['time'].dt.day_of_week

for idx in range(len(list_of_points)):
    print(idx)
    m = folium.Map(location=TALLINN_LAT_LONG, tiles='openstreetmap', zoom_start=13)
    for point_idx, point in enumerate(list_of_points[idx]):
        # Add circle bubbles to the map (x - Lat, y - Long)
        # folium.Marker([row['geometry'].x, row['geometry'].y]).add_to(m)
        folium.Circle(
            location=[point.x, point.y],
            radius=CIRCLE_RADIUS * list_of_vals[idx][point_idx],
            color='green'
        ).add_to(m)
        folium.map.Marker(
            TALLINN_LAT_LONG,
            icon=DivIcon(
                icon_size=(250, 36),
                icon_anchor=(0, 0),
                html=f'<div style="font-size: 20pt">Day {days_of_week[idx]} hour {hours[idx]} </div>',
            )
        ).add_to(m)
    image_data = m._to_png(5)
    image = Image.open(io.BytesIO(image_data))
    image.save(f'map_{idx}.png')

image_filenames = glob.glob('./*.png')
images = []
for image_filename in image_filenames:
    images.append(imageio.imread(image_filename))
imageio.mimsave('map.gif', images, fps=2)
