#!/usr/bin/env python
# coding: utf-8

# # Geospatial data

# ### Getting geospatial data from open sources

# Querying points of interest (POIs) using Overpass API. Overpass API is OSM’s querying API. It is incredibly powerful in that it can very quickly return queried features, and allows for selection of location, tags, proximity, and more.
# 
# Let’s query for restaurants near the University of Toronto.

# In[79]:


get_ipython().system('pip install overpass')


# In[80]:


import overpass

api = overpass.API()

# We're looking for restaurants within 1000m of a given point
overpass_query = """
(node["amenity"="restaurant"](around:1000,43.66, -79.39);
 way["amenity"="restaurant"](around:1000,43.66, -79.39);
 rel["amenity"="restaurant"](around:1000,43.66, -79.39);
);
out center;
"""

restaurants = api.get(overpass_query)


# The example above uses the overpass package, which by default returns results in geojson format. See the <a href="https://github.com/mvexel/overpass-api-python-wrapper", target="new">overpass documentation</a> for more information.
# 
# 
# Next, let’s extract some data about each restaurant, and then plot all of them on a map. This time, we’ll use a plotly ScatterMapBox, which uses tiles from MapBox. You can refer to plotly’s documentation here. Each POI on that map has a tooltip that shows the restaurant’s name when hovered.

# In[81]:


get_ipython().system('pip install plotly')
import plotly.graph_objects as obj

# Extract the lon, lat and name of each restaurant:
coords = []
text = []
for elem in restaurants['features']:
    latlon = elem['geometry']['coordinates']
    if latlon == []: continue
    coords.append(latlon)
    if 'name'  not in elem['properties']:
        text.append('NONAME')
    else:
        text.append(elem['properties']['name'])
        
# Convert into a dictionary for plotly
restaurant_dict = dict(type='scattermapbox',
                   lat=[x[1] for x in coords], 
                   lon=[x[0] for x in coords],
                   mode='markers',
                   text=text,
                   marker=dict(size=8, color='blue'),
                   hoverinfo='text',    
                   showlegend=False)


# plotting restaurants' locations around University of Toronto

center=(43.662643, -79.395689) # UofT main building

fig = obj.Figure(obj.Scattermapbox(restaurant_dict))

# defining plot layout
fig.update_layout(mapbox_style="stamen-terrain")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, mapbox = {'center': {'lat': center[0], 'lon': center[1]}, 'zoom': 13})
fig.show()


# You can compile your own dataset using the Overpass QL language that runs on Overpass turbo. You can use this query language to mine OpenStreetMaps data, filter it, and get it ready to be used by osmnx or any library that parses .osm files. Below is a quick review about using Overpass API, which is the official API for reading data from OpenStreetMap servers. All the online routing services and software use it. Additionally, we will usually use Nominatim to do geocoding/geo-decoding; translating addresses to/from (latitude-longitude).
# 
# Also be aware of the fact that most of the time if you are building a dataset over a very big area in the map, the graph parsed from the data by osmnx won’t be complete, even though there are physically feasible routes that could make the graph complete and connect all the nodes. This deficiency is usually because of the incomplete relations and data of osm.

# ### Getting data using OverPass QL
# Fire up Overpass turbo and run these scripts and export it as .osm files.
For all hospitals around UofT, write:
    
[bbox:43.611968,-79.464798,43.695183,-79.297256];
node["amenity"="hosptial"]({{bbox}});
out;
way["amenity"="hospital"]({{bbox}});
out center;
relation["amenity"="hospital"]({{bbox}});
out center;  

FOr All Tim Hortons in Canada, write:
[bbox:42.553080,-141.328125,69.960439,-51.679688];
node["amenity"="cafe"]({{bbox}});
node["name"="Tim Hortons"]({{bbox}});
out center;

For all fast food or restaurant places in London write:
area
  ["boundary"="administrative"]
  ["name"="London"]->.a;          // Redirect result to ".a"
out body qt;
(
  node
    (area.a)                    // Use result from ".a"
    ["amenity"~"fast_food|restaurant"];
  way
    (area.a)                    // Use again result from ".a"
    ["amenity"~"fast_food|restaurant"];
);
out body qt;
>;
out skel qt;

# Finding the bounding box around an area of interest is a recurring problem in writing OverPass QL queries. To solve for that, we can use bbox finder. Don’t forget to change the coordinate format to latitude/longitude at the right corner after drawing the polygon around the area of interest.

# ### Getting data using Overpass turbo’s Wizard
# 
# <a href="https://overpass-turbo.eu/">Overpass turbo</a>’s Wizard provides an easy way to auto-generate Overpass QL queries. Wizard syntax is similar to that of a search engine. An example of Wizard syntax is amenity=hospital that generates an Overpass QL query to find all the hospitals in a certain region of interest. Hospital locations will be visualized on the map and can be downloaded/copied using the “Export” button. The data can be exported as GeoJSON, GPX, KML, raw OSM data, or raw data directly from Overpass API. You can then use osmnx to read .osm files with osmnx.graph_from_xml.
Some examples of Overpass turbo’s wizard syntax include:

amenity=restaurant in "Toronto, Canada" to find all resturants in City of Toronto.

amenity=cafe and name="Tim Hortons" to find all Tim Hortons coffee shops.

(amenity=hospital or amenity=school) and (type:way) to find hospitals and schools with close ways mapped.

amenity=hospital and name~"General Hospital" will find all hospitals with “General Hospital” part of their names.
# ### Calculating haversine distance between two points of interest

# In[82]:


# coordinates of two points in (latitude, longitude) format
LA = (34.052235, -118.243683) # Los Angeles, USA 
Madrid = (40.416775, -3.703790) # Madrid, Spain


# #### Calculating haversine distance using math

# In[83]:


from math import radians, sin, cos, sqrt, atan2

# convert coordinates to radians
lat1, lon1 = radians(LA[0]), radians(LA[1])
lat2, lon2 = radians(Madrid[0]), radians(Madrid[1])

# calculate haversine distance
a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
c = 2 * atan2(sqrt(a), sqrt(1 - a))
distance = 6371 * c  # 6371 is the radius of the Earth in kilometers

print(distance)


# #### Calculating haversine distance using haversine library

# In[84]:


get_ipython().system('pip install haversine')
from haversine import haversine

# calculate the distance in kilometers
distance = haversine(LA, Madrid)
print(distance)


# #### Calculating haversine distance using geopy

# In[85]:


get_ipython().system('pip install geopy')


# In[86]:


from geopy.distance import geodesic

print(geodesic(LA,Madrid).km)


# In[87]:


from geopy.distance import geodesic
from geopy.geocoders import Nominatim

def haversine_distance(city1, city2):
    # Coordinates of the two cities (latitude, longitude)
    coords_city1 = city1.latitude, city1.longitude
    coords_city2 = city2.latitude, city2.longitude
    
    # Calculate the Haversine distance
    distance = geodesic(coords_city1, coords_city2).kilometers
    return distance

# Initialize the geocoder
geolocator = Nominatim(user_agent="city_distance_app")

# Get the coordinates of the cities
city1 = geolocator.geocode("Los Angeles")
city2 = geolocator.geocode("Madrid")

# Calculate the Haversine distance
distance = haversine_distance(city1, city2)
print(f"Haversine Distance between {city1.address} and {city2.address}: {distance:.2f} kilometers")


# #### Calculating haversine distance using sklearn

# In[88]:


get_ipython().system('pip install sklearn')


# In[89]:


from sklearn.metrics.pairwise import haversine_distances
from math import radians

# convert coordinates to radians
LA_in_radians = [radians(_) for _ in LA]
Madrid_in_radians = [radians(_) for _ in Madrid]

distance = haversine_distances([LA_in_radians, Madrid_in_radians])
distance=distance* 6371000/1000  # multiply by Earth radius to get kilometers

print(distance)


# ### Handling data using geopandas

# In[90]:


get_ipython().system('pip install geopandas')

import geopandas as gpd

# file downloaded from https://data.ontario.ca/dataset/ontario-s-health-region-geographic-data
ontario = gpd.read_file(r"data/OntarioHealth/Ontario_Health_Regions.shp")
ontario = ontario.to_crs('EPSG:4326')
ontario = ontario[(ontario.REGION != "North")] # exclude Northern Ontario

ontario


# ### Getting elevation Data

# In[ ]:


# Let’s first get the centroids for each region:
ontario['centroid'] = ontario.centroid
ontario


# Notice that calculating the centroid raises a warning. That’s because we are using EPSG:4326, which uses degrees as a unit of measure. This makes polygon calculations inaccurate, especially at larger scales. We will ignore this warning for this example, but keep in mind that centroids will not be accurate in this projection. For better results, you can calculate the centroids of a projection that uses a flat projection (that retains area) and then reproject it back to EPSG:4326.
# 
# For web use, when the desired effect is a visual centroid, it is possible to continue using a Mercator projection like EPSG:4326, while applications that require a “true” centroid should use a projection like Equal Area Cylindrical, which avoids distortion at the poles. See here for more details.

# In[ ]:


# Now, let’s query the Open Elevation API for the elevation (in metres) at the centroids for each region.
from requests import get

def get_elevation(centroid):
    query = (f'https://api.open-elevation.com/api/v1/lookup?locations={centroid.y},{centroid.x}')

    # Set a timeout on the request in case of a slow response
    r = get(query,timeout=30)

    # Only use the response if the status is successful
    if r.status_code!=200 and r.status_code!=201: return None

    elevation = r.json()['results'][0]['elevation']
    return elevation

elevations = []

for index, row in ontario.iterrows():
    elevations.append(get_elevation(row['centroid']))
    
ontario['elevations'] = elevations
ontario


# In[ ]:


get_ipython().system('pip install folium')


# In[ ]:


# let's visualzie this data
import geopandas as gpd

# Load Ontario dataset if available
ontario = gpd.read_file("your_file.geojson")  # Change filename if necessary
print(ontario.head())

import folium

# Set starting location, initial zoom, and base layer source.
m = folium.Map(location=[43.67621,-79.40530],zoom_start=7, tiles='cartodbpositron', scrollWheelZoom=False, dragging=True)

for index, row in ontario.iterrows():
    # Simplify each region's polygon as intricate details are unnecessary
    sim_geo = gpd.GeoSeries(row['geometry']).simplify(tolerance=0.001)
    geo_j = sim_geo.to_json()
    geo_j = folium.GeoJson(data=geo_j, name=row['REGION'],style_function=lambda x: {'fillColor': 'black'})
    folium.Popup(row['REGION']).add_to(geo_j)
    geo_j.add_to(m)

for index, row in ontario.iterrows():
    folium.Marker(location=[row['centroid'].y,row['centroid'].x], popup='Elevation: {}'.format(row['elevations'])).add_to(m)
m


# ### Fetching OpenStreetMap data using osmnx

# In[ ]:


get_ipython().system('conda create -n ox -c conda-forge --strict-channel-priority osmnx')


# In[ ]:


import osmnx as ox
import matplotlib.pyplot as plt


place_name = "Plaza de España, Madrid"

# fetch OSM street network (drive, walk, private, etc.) from the location
graph = ox.graph_from_address(place_name, network_type='walk')
fig, ax = ox.plot_graph(graph)


# #### Edges and Node
# 
# We can extract the nodes and edges of the graph as separate structures.

# In[ ]:


nodes, edges = ox.graph_to_gdfs(graph)

nodes.head(5)


# In[ ]:


edges.head(5)


# We can further drill down to examine each individual node or edge.

# In[91]:


# Rendering the 2nd node
list(graph.nodes(data=True))[1]


# In[92]:


# Rendering the 1st edge
list(graph.edges(data=True))[0]


# #### Street Types
# Street types can also be retrieved for the graph:

# In[93]:


print(edges['highway'].value_counts())


# #### Calculating Network Statistics

# In[94]:


ox.basic_stats(graph)


# We can also see the circuity average. Circuity average is the sum of edge lengths divided by the sum of straight line distances. It produces a metric > 1 that indicates how “direct” the network is (i.e. how much more distance is required when travelling via the graph as opposed to walking in a straight line).

# In[95]:


# osmnx expects an undirected graph
undir = graph.to_undirected()
ox.stats.circuity_avg(undir)


# #### Extended and Density Stats

# In[96]:


convex_hull = edges.unary_union.convex_hull
convex_hull


# #### CRS Projection
# You can also look at the projection of the graph. To find out more about projections, check out this section. Additionally, you can also reproject the graph to a different CRS.

# In[97]:


edges.crs


# In[98]:


merc_edges = edges.to_crs(epsg=3857)
merc_edges.crs


# #### Shortest Path Analysis
# 
# To calculate the shortest path, we first need to find the closest nodes on the network to our starting and ending locations.

# In[99]:


get_ipython().system('pip install ipyleaflet')


# In[100]:


from ipyleaflet import *

# Plaza Mayor Madrid
center=(40.4155, -3.7043) 
# Puerta del Sol
source_point = (40.416729, -3.703339)  
# Plaza de España
destination_point = (40.423382, -3.712165) 

m = Map(center=center, zoom=14)
m.add_layer(Marker(location=source_point, icon=AwesomeIcon(name='camera', marker_color='red')))
m.add_layer(Marker(location=center, icon=AwesomeIcon(name='graduation-cap')))
m.add_layer(Marker(location=destination_point, icon=AwesomeIcon(name='university',marker_color='green')))
m


# Notice that the way we create maps in ipyleaflet is different from folium. For the latter, the code is as follows:

# In[101]:


import folium
m = folium.Map(location=center, zoom_start=15)
folium.Marker(location=source_point,icon=folium.Icon(color='red',icon='camera', prefix='fa')).add_to(m)
folium.Marker(location=center,icon=folium.Icon(color='blue',icon='graduation-cap', prefix='fa')).add_to(m)
folium.Marker(location=destination_point,icon=folium.Icon(color='green',icon='university', prefix='fa')).add_to(m)
m


# Let’s revisit our trip across Madrid from the statue in Puerta del Sol to Plaza de España.
# 
# To calculate the shortest path, we first need to find the closest nodes on the network to our starting and ending locations.

# In[102]:


get_ipython().system('pip install geopandas')


# In[103]:


import geopandas

X = [source_point[1], destination_point[1]]
Y = [source_point[0], destination_point[0]]
closest_nodes = ox.distance.nearest_nodes(graph,X,Y)

# Get the rows from the Node GeoDataFrame
closest_rows = nodes.loc[closest_nodes]

# Put the two nodes into a GeoDataFrame
od_nodes = geopandas.GeoDataFrame(closest_rows, geometry='geometry', crs=nodes.crs)
od_nodes


# Let’s find and plot the shortest route now!

# In[ ]:


get_ipython().system('pip install networkx[default]')


# In[ ]:


import networkx

shortest_route = networkx.shortest_path(G=graph,source=closest_nodes[0],target=closest_nodes[1], weight='length')
print(shortest_route)


# In[ ]:


ox.plot_graph_route(graph,shortest_route,figsize=(15,15))


# Let’s make a map that shows the above route, with both starting and ending nodes shown as markers using draw_route implemented as part of our Python package optalgotools.

# In[ ]:


get_ipython().system('pip install optalgotools')


# In[ ]:


from optalgotools import routing

routing.draw_route(graph, shortest_route)


# #### Retrieve buildings from named place
# 
# Just like our graph above, we can also retrieve all the building footprints of a named place.

# In[ ]:


# Retrieve the building footprint, project it to match our previous graph, and plot it.
buildings = ox.geometries.geometries_from_address(place_name, tags={'building':True}, dist=300)
buildings = buildings.to_crs(edges.crs)
ox.plot_footprints(buildings)


# Now that we have the building footprints of Madrid, let’s plot that shortest route again.
# 
# First, get the nodes from the shortest route, create a geometry from it, and then visualize building footprints, street network, and shortest route all on one plot.
# 

# In[ ]:


get_ipython().system('pip install shapely')


# In[ ]:


from shapely.geometry import LineString

# Nodes of our shortest path
route_nodes = nodes.loc[shortest_route]

# Convert the nodes into a line geometry
route_line = LineString(list(route_nodes.geometry.values))


# In[ ]:


# Create a GeoDataFrame from the line
route_geom = geopandas.GeoDataFrame([[route_line]], geometry='geometry', crs=edges.crs, columns=['geometry'])

# Plot edges and nodes
ax = edges.plot(linewidth=0.75, color='gray', figsize=(15,15))
ax = nodes.plot(ax=ax, markersize=2, color='gray')

# Add building footprints
ax = buildings.plot(ax=ax, facecolor='khaki', alpha=0.7)

# Add the shortest route
ax = route_geom.plot(ax=ax, linewidth=2, linestyle='--', color='red')

# Highlight the starting and ending nodes
ax = od_nodes.plot(ax=ax, markersize=100, color='green')


# ### Fetching data using pyrosm

# In[ ]:


get_ipython().system('pip install pyrosm')


# In[ ]:


import pyrosm
import matplotlib

# List available places
available_places = pyrosm.data.available
print(available_places.keys())
print(available_places['cities'])


# In[ ]:


place_name = 'Cairo'
file_path = pyrosm.get_data(place_name)
print('Data downloaded to:', file_path)


# In[ ]:


# Initialises the OSM object that parses the generated .osm.pbf files
osm = pyrosm.OSM(file_path)
print('osm type:', type(osm))


# ### Getting data from open data repositories
# 
# This is an example of how to read data from URL. The Bike Share Toronto Ridership data contains anonymized trip data, including: Trip start day and time, Trip end day and time, Trip duration, Trip start station, Trip end station, User type. This dataset is from Toronto Parking Authority, published on https://open.toronto.ca/dataset/bike-share-toronto-ridership-data/.

# In[ ]:


import requests
import json
import matplotlib.pyplot as plt
import folium

# URL of the Bike Share dataset file
url = 'https://tor.publicbikesystem.net/ube/gbfs/v1/en/station_information'

# Fetch the contents of the URL
response = requests.get(url)

# Extract the JSON content from the response
json_data = response.json()

# Extract the station informations
stations = json_data.get("data", {}).get("stations", [])
extracted_data = []
for station in stations:
  extracted_data.append({"station_id": station.get("station_id"), "name": station.get("name"), "lat": station.get("lat"), "lon": station.get("lon"), "address": station.get("address"), "capacity": station.get("capacity")})

# Create a folium map centered around Toronto
toronto_map = folium.Map(location=[43.651070, -79.347015], zoom_start=12)

# Add markers for each bike share station
for station in extracted_data:
    lat = station["lat"]
    lon = station["lon"]
    id = station["station_id"]
    name = station["name"]
    capacity = station["capacity"]

    # Create a marker and add it to the map
    folium.Marker(
        location=[lat, lon],
        popup=f"Id: {id}<br>Name: {name}<br>Capacity: {capacity}",
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(toronto_map)

# Display the map
toronto_map


# ### Download the Microsoft building footprints using leafmap

# In[ ]:


get_ipython().system('pip install leafmap')


# In[ ]:


import leafmap

country = "Canada"

# Specify the number of files to download. Set to None to download all files.
head = 2

leafmap.download_ms_buildings(
    country, 
    out_dir="buildings", 
    merge_output=f"{country}_ms.shp", 
    head=head
    )


# ### Display the building footprints

# In[ ]:


m = leafmap.Map()
m.add_basemap("SATELLITE")
m.add_vector(f"{country}_ms.shp", layer_name="MS Buildings")
m


# ***
# 
# # Geospatial Datasets
# 
# This section includes a non-exhaustive list of datasets that may be of interest to you. Some sources are open source, while others offer "freemium" or paid options.
# 
# #### 1. Available open datasets
# * [Geofabrik](https://download.geofabrik.de/index.html)
# * [Factory POI](http://www.poi-factory.com/)
# * [Global Rural-Urban Mapping Project (GRUMP)](https://sedac.ciesin.columbia.edu/data/set/grump-v1-settlement-points)
# * [GeoNames Data](https://www.geonames.org/export/)
# * [City of Toronto](https://www.toronto.ca/city-government/data-research-maps/open-data/)
# * [ArcGIS Hub](https://www.esri.com/en-us/arcgis/products/arcgis-hub/overview)
# * [GeoHub City of Brampton](https://geohub.brampton.ca/pages/data)
# * [City of Markham](https://data-markham.opendata.arcgis.com/)
# * [New York City](https://opendata.cityofnewyork.us/)
# * [BBBike](https://extract.bbbike.org/)
# * [Mapzen](https://github.com/tilezen/joerd/tree/master/docs)
# * [openterrain list](https://github.com/openterrain/openterrain/wiki/Terrain-Data)
# * [terrain party](https://terrain.party/)
# * [OpenMapTiles](https://openmaptiles.org/)
# * [UofT MDL](https://mdl.library.utoronto.ca/)
# * [GADM maps and data](https://gadm.org/index.html)
# * [Elevation data](https://www.opentopodata.org/)
# * [SRTM C-BAND DATA PRODUCTS](https://www2.jpl.nasa.gov/srtm/cbanddataproducts.html)
# * [CGIAR-CSI SRTM raster data](https://srtm.csi.cgiar.org/srtmdata/)
# * [Wikimapia](https://wikimapia.org/)
# * [geodatasets](https://geodatasets.readthedocs.io/en/latest/)
# * [ Countries States Cities Database](https://github.com/dr5hn/countries-states-cities-database)
# * [Hugging Face Geospatial Data](https://huggingface.co/datasets?modality=modality:geospatial&sort=trending)
# 
# ---
# 
# #### 2. Commercially available datasets
# * [Mapbox](https://www.mapbox.com/data-products)
# * [Planet.osm](https://planet.openstreetmap.org/)
# * [MapTiler](https://www.maptiler.com/)
# * [Factual Global Places](https://www.factual.com/data-set/global-places/)
# * [TravelTime API](https://docs.traveltime.com/api/overview/introduction)
# * [Precisely](https://www.precisely.com/)
# * [World Cities Database](https://www.worldcitiesdatabase.com )
# * [SafeGraph](https://www.safegraph.com/)
# * [Google Maps Platform](https://cloud.google.com/maps-platform/)
# * [Python Client for Google Maps Services](https://github.com/googlemaps/google-maps-services-python)
# * [Here Maps for Developers](https://developer.here.com/products/here-sdk)
# * [Ratio City](https://www.ratio.city/)
# * [100 feet](https://www.beans.ai/index)
# * [MPAC Residential Property Assessments](https://www.mpac.ca/)
# * [Geodata Tufts](https://geodata.tufts.edu/)
# * [TomTom API](https://developer.tomtom.com/)
# 
# ---
# 
# #### 3. Elevation Data
# * [Open Elevation API](https://open-elevation.com/)
# * [Open Topo Data API](https://www.opentopodata.org/#public-api)
# * [National Map API](https://nationalmap.gov/)
# * [Google Maps API](https://developers.google.com/maps/documentation/elevation/overview)
# 
# ---
# 
# #### 4. Traffic Datasets
# * [traffic per edge](https://github.com/Project-OSRM/osrm-backend/wiki/Traffic)
# * [Open Traffic](https://github.com/opentraffic)
# * [Google Routes](https://cloud.google.com/maps-platform/routes)
# * [Chicago Traffic Tracker](https://chicagotraffictracker.com/)
# 
# ---
# #### 5. Parking Tickets Datasets
# * [City of Toronto Parking Tickets](https://ckan0.cf.opendata.inter.prod-toronto.ca/tr/dataset/parking-tickets)
# * [Toronto Parking Tickets Visualziation](https://github.com/ian-whitestone/toronto-parking-tickets)
# * [Others](https://data.world/datasets/parking-ticket)
# 
# ---
# #### 6. Public Transport Networks Datasets
# * [Google Transit APIs](https://developers.google.com/transit)
# 
# ---
# #### 7. Planned events, road work, and other temporary changes to the road network and bridge, tunnel and ferry events
# * [one.network](https://us.one.network/)
# * [ROAD dataset](https://github.com/gurkirt/road-dataset)
# * [Road Point Events](https://open.canada.ca/data/en/dataset/35e1d8d3-cb2f-434d-a20c-584ea5037fa0)
# 
# ---
# #### 8. Traffic Crashes Datasets
# * [A Countrywide Traffic Accident Dataset (2016 - 2020)](https://www.kaggle.com/sobhanmoosavi/us-accidents)
# * [traffic per edge](https://github.com/Project-OSRM/osrm-backend/wiki/Traffic) 
# * [Open traffic](https://github.com/opentraffic)
# * [AV Documented Incidents](https://www.austintexas.gov/page/autonomous-vehicles)
# * [Others](https://data.world/datasets/crash)
# 
# ---
# #### 9. Emission Datasets
# * [Open Climate Data](https://openclimatedata.net/)
# * [Ontario Air Quality Data Sets](http://www.airqualityontario.com/science/data_sets.php)
# * [Others](https://data.world/datasets/co2)
# 
# ---
# #### 10. Environmental Datasets
# * [Canadian Open Geospatial Data](https://canadiangis.com/data.php)
# * [Government of Canada Open Data Portal](https://open.canada.ca/data/en/dataset)
# 
# ---
# #### 11. Mobility-aware urban design, active transportation modeling and access analysis for amenities and public transport
# * [Urbano](https://www.urbano.io/)
# 
# 
# ---
# #### 12. Accessibility
# * [Wheelmap](https://wheelmap.org/)
# * [accessibility.cloud](https://www.accessibility.cloud/)
# 
# ---
# #### 13. Crime map
# * [Crime map](https://www.crimemapping.com/map/agency/91)
# * [Others](https://data.world/datasets/crime)
# 
# ---
# #### 14. Open Source Projects
# * [GIScience](https://github.com/GIScience)

# In[ ]:


import pandas as pd

# Load the dataset (Ensure correct filename)
df = pd.read_csv("Listing-B.3_Geospatial_dataset.csv")

# Display first few rows
print("Dataset Loaded Successfully!")
print(df.head())  # Check first 5 rows


# In[ ]:


import folium

# Create a map centered on Saudi Arabia
map_saudi = folium.Map(location=[23.8859, 45.0792], zoom_start=5)

# Add city markers
for _, city in df.iterrows():
    folium.Marker([city['Latitude'], city['Longitude']], popup=city['City']).add_to(map_saudi)

# Display map
map_saudi


# In[ ]:


import networkx as nx

# Create a graph for cities
G = nx.Graph()

# Add weighted edges (approximate distances between cities)
for i, city1 in df.iterrows():
    for j, city2 in df.iterrows():
        if i != j:
            distance = ((city1['Latitude'] - city2['Latitude'])**2 + (city1['Longitude'] - city2['Longitude'])**2)**0.5 * 111
            if distance < 1000:  # Only connect cities within 1000 km
                G.add_edge(city1['City'], city2['City'], weight=distance)

print("Graph Created Successfully!")


# In[ ]:


# Define Start and End Cities
start_city = "Riyadh"
end_city = "Jeddah"

# BFS
bfs_edges = list(nx.bfs_edges(G, start_city))
bfs_nodes = [start_city] + [edge[1] for edge in bfs_edges]

# DFS
dfs_edges = list(nx.dfs_edges(G, start_city))
dfs_nodes = [start_city] + [edge[1] for edge in dfs_edges]

# Dijkstra's Shortest Path
dijkstra_path = nx.shortest_path(G, source=start_city, target=end_city, weight='weight')

# Print Results
print("BFS Path:", " → ".join(bfs_nodes))
print("DFS Path:", " → ".join(dfs_nodes))
print("Dijkstra's Shortest Path:", " → ".join(dijkstra_path))


# In[ ]:


import matplotlib.pyplot as plt

# Plot the Network Graph
plt.figure(figsize=(6, 4))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', font_size=10, node_size=3000)
plt.title("Saudi Arabia Road Network")
plt.show()


# In[107]:


import pandas as pd
import folium
import networkx as nx
import random
from queue import Queue
from IPython.display import display

# Load the dataset from a CSV file.
df = pd.read_csv('worldcities.csv')

# Filter the dataset to only include the selected cities
selected_cities = ["Riyadh", "Dhahran", "Arar"]
df = df[df['city'].isin(selected_cities)]

# Create a dictionary mapping each city to its (latitude, longitude) coordinates
cities = {row['city']: (row['lat'], row['lng']) for _, row in df.iterrows()}

# Create a Folium map centered on Riyadh
saudi_map = folium.Map(location=cities["Riyadh"], zoom_start=6)
for city, coords in cities.items():
    folium.Marker(location=coords, popup=city).add_to(saudi_map)

# Build a simple graph representing a road network (distances in km)
G = nx.Graph()
for city, coords in cities.items():
    G.add_node(city, pos=coords)

# Define weighted edges between the cities with approximate distances (in km)
edges = [
    ("Riyadh", "Dhahran", 400),
    ("Dhahran", "Arar", 900),
    ("Riyadh", "Arar", 1100),
]
G.add_weighted_edges_from(edges)

# Step 5: Implement BFS, DFS, Dijkstra, and Simulated Annealing for route finding
def bfs_path(graph, start, goal):
    queue = Queue()
    queue.put([start])
    while not queue.empty():
        path = queue.get()
        node = path[-1]
        if node == goal:
            return path
        for neighbor in graph.neighbors(node):
            if neighbor not in path:
                new_path = list(path)
                new_path.append(neighbor)
                queue.put(new_path)
    return None

def dfs_path(graph, start, goal, path=None):
    if path is None:
        path = [start]
    if start == goal:
        return path
    for neighbor in graph.neighbors(start):
        if neighbor not in path:
            new_path = dfs_path(graph, neighbor, goal, path + [neighbor])
            if new_path:
                return new_path
    return None

# Dijkstra's algorithm for shortest path
def dijkstra_path(graph, start, goal):
    return nx.shortest_path(graph, source=start, target=goal, weight='weight')

# Simulated Annealing (Simplified version for demonstration)
def simulated_annealing_path(graph, start, goal):
    current_path = list(graph.nodes)
    random.shuffle(current_path)
    return current_path[:current_path.index(goal)+1]

# Compare the algorithms
origin = "Riyadh"
destination = "Dhahran"

bfs_result = bfs_path(G, origin, destination)
dfs_result = dfs_path(G, origin, destination)
dijkstra_result = dijkstra_path(G, origin, destination)
simulated_annealing_result = simulated_annealing_path(G, origin, destination)

print("BFS Path:", bfs_result)
print("DFS Path:", dfs_result)
print("Dijkstra's Path:", dijkstra_result)
print("Simulated Annealing Path:", simulated_annealing_result)

# Save and display the map
saudi_map.save("saudi_map.html")
display(saudi_map)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




