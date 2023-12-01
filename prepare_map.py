import matplotlib.pyplot as plt
import rioxarray as rxr
import numpy as np
from PIL import Image
from skimage.transform import resize
import json
from pyproj import Transformer
import shapefile
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Polygon

def remove_outliers(data, sigma=4):
        std = np.std(data)
        mean = np.mean(data)
        data = np.array([ d if abs(d - mean) < sigma * std else mean for d in data ])
        return data

def export_raster(code):
    tif_file = f'data/{code}.tif'
    png_file = f'data/{code}.png'
    output_path = f'output/dem/{code}_dem.png'
    
    dsm = rxr.open_rasterio(tif_file, masked=True).squeeze()
    dsm_array = np.array(dsm)[:-1, 1:]

    with Image.open(png_file) as image:
        w, h = image.size

    dsm_array = resize(dsm_array, (h, w))
    # dsm_array = remove_outliers(dsm_array.ravel()).reshape(dsm_array.shape)
    plt.imsave(output_path, dsm_array, cmap='gray', vmin=dsm_array.min(), vmax=dsm_array.max() )
    np.save(f'output/dem/{code}.npy', dsm_array)
    print(f'{output_path} are saved')

def get_buildings(code):
    def covert_coordinate(array, X1, Y1, ratio_w, ratio_h, h):
        for a in array:
            a[0] = round((a[0] - X1) * ratio_w)
            a[1] = h - round((a[1] - Y1) * ratio_h)
        return array.ravel()

    def getFeatures(gdf):
        return json.loads(gdf.to_crs(4326).to_json())['features'][0]['geometry']['coordinates'][0]

    tags = {'building': True}
    # tags = {'drive': True}
    transformer = Transformer.from_crs("EPSG:6344", "EPSG:4326")
    shape_file = f'data/{code}.shp'
    png_file = f'data/{code}.png'
    
    X1, Y1, X2, Y2 = shapefile.Reader(shape_file).bbox
    latitude, longitude = (X1 + X2) / 2, (Y1 + Y2) / 2 
    latitude, longitude = transformer.transform(latitude, longitude)
    buildings = ox.features_from_point((latitude, longitude), tags)
    
    # streets_graph = ox.graph_from_point((latitude, longitude), network_type='drive')
    # streets_graph = ox.projection.project_graph(streets_graph)
    # buildings = ox.graph_to_gdfs(streets_graph, nodes=False, edges=True,
    #                                node_geometry=False, fill_edge_geometry=True)
        
    df = gpd.read_file(shape_file)
    df.crs = "EPSG:6344"
    polygon = Polygon(getFeatures(df))
    buildings = ox.features_from_polygon(polygon, tags)
    buildings = buildings.to_crs('EPSG:6344')

    with Image.open(png_file) as image:
        w, h = image.size
        ratio_w = w / (X2 - X1)
        ratio_h = h / (Y2 - Y1)

    with open('template.json') as f:
        data = json.load(f)

    for i, f in enumerate([f'{code}.png', f'{code}_dem.png']):
        image_dict = {
            "id": i + 1,
            "width": w,
            "height": h,
            "file_name": f,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        } 

        data['images'].append(image_dict)
    
    for i, g in enumerate(buildings.geometry):
        array = np.array(g.exterior.coords)
        array = covert_coordinate(array, X1, Y1, ratio_w, ratio_h, h)
    
        array = [ a for i , a in enumerate(array) if a > 0 or a < image.size[i%2] ]
        if len(array) == 0:
            continue

        annotation = {
            "id": i,
            "image_id": 2,
            "category_id": 2,
            "segmentation": [array],
            "area": g.area,
            "bbox": g.bounds,
            "iscrowd": 0,
            "attributes": {
                "occluded": False
            }
        }

        data['annotations'].append(annotation)

    with open(f"output/json/{code}_buildings.json", "w") as f:
        json.dump(data, f)

    fig = plt.figure(dpi=500, frameon=False)
    ax = buildings.plot()
    ax.set_xlim(X1, X2)
    ax.set_ylim(Y1, Y2)
    ax.set_axis_off()
    ax.figure.savefig(f'output/json/{code}.png', bbox_inches='tight', pad_inches=0)
    print(f'output/json/{code}.png are saved')

if __name__ == '__main__':
    code_list = ['A4', 'A5', 'A6', 'B4', 'B5', 'B6', 'C4', 'C5', 'C6', 'C7', 'D4', 'D5', 'D6', 'D7']
    code_list = ['E4', 'E5', 'E6', 'E7']

    for code in code_list:
        export_raster(code)
        get_buildings(code)