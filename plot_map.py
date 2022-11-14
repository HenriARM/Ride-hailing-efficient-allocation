import pandas as pd
import geopandas as gpd
import folium
from PIL import Image
import io
import imageio

# from folium import Choropleth, Circle, Marker
# from folium.plugins import HeatMap, MarkerCluster
# import matplotlib.pyplot as plt
# import webbrowser

f = "./robotex5.csv"
df = pd.read_csv(f)
df = df.sort_values(by=["start_time"])

df = df[:100]
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['start_lat'], df['start_lng']))
print(gdf.info())
print(gdf.head())

# https://www.latlong.net/place/tallinn-estonia-5882.html
TALLINN_LAT_LONG = [59.436962, 24.753574]
m = folium.Map(location=TALLINN_LAT_LONG, tiles='openstreetmap', zoom_start=13)

# TODO: print hour by hour (check how to real csv string to datetime)
#  and generate .gif https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
# Add circle bubbles to the map
CIRCLE_RADIUS = 20
for idx, row in gdf.iterrows():
    # x - Lat, y - Long
    # folium.Marker([row['geometry'].x, row['geometry'].y]).add_to(m)
    folium.Circle(
        location=[row['geometry'].x, row['geometry'].y],
        radius=CIRCLE_RADIUS * row['ride_value'],
        color='green'  # 'red'
    ).add_to(m)

# m.save('map.html')
img_data = m._to_png(5)
img = Image.open(io.BytesIO(img_data))
map_fn = f'map.png'
img.save(map_fn)

# images = []
# for filename in filenames:
#     images.append(imageio.imread(filename))
# imageio.mimsave('map.gif', images)

# TODO: if point goes out of view?

# # one option https://www.diva-gis.org/datadown
# data = gpd.read_file("./EST_adm/EST_adm1.shp")
# data.plot()
# plt.show()
# # and then video https://github.com/bendoesdata/make-a-gif-map-in-python/blob/master/make-a-gif-map/make-a-gif-map.ipynb
