import plotly.graph_objects as go
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import rasterio
from rasterio.plot import show
from rasterio.merge import merge
import laspy
import open3d
import os
import pandas as pd

import rasterio
import pandas as pd
import math


from pylab import *
import datetime

import time
import rioxarray as rxr
import numpy as np
import xarray as xr
from rasterio.crs import CRS

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib import dates

from pycocotools.coco import COCO
from matplotlib import image
from pathlib import Path

import cv2
import json
from skimage.transform import resize

def main():
    las = laspy.read("./tnris/stratmap18-50cm_2995222a1.laz")
    point_data_1 = np.stack([las.x, las.y, las.z], axis=0).transpose((1, 0))

    las = laspy.read("./tnris/stratmap18-50cm_2995222a3.laz")
    point_data_2 = np.stack([las.x, las.y, las.z], axis=0).transpose((1, 0))

    point_data = np.vstack((point_data_1, point_data_2))

    header = laspy.header.Header()
    outfile = laspy.file.File("output.las", mode="w", header=header)
    outfile.close()
    
    # geom = open3d.geometry.PointCloud()
    # geom.points = open3d.utility.Vector3dVector(point_data)
    # open3d.visualization.draw_geometries([geom])

def read_img():
    dir = 'tnris'
    filename_list = ['stratmap18-1m_2995222a1.img', 'stratmap18-1m_2995222a3.img']

    rester_list = []
    for f in filename_list:
        rester_list.append(rasterio.open(os.path.join(dir, f)))
        print(rasterio.open(os.path.join(dir, f)).profile)
        show(rasterio.open(os.path.join(dir, f)))

    mosaic, output = merge(rester_list, bounds=polygon.bounds)
    show(mosaic)

def find_file():
    dir = 'tnris'
    coord_list = ['westbc', 'eastbc', 'northbc', 'southbc']
    target_list = [(-95.243, 29.722), (-95.244, 29.715)]
    
    for target in target_list:
        
        for f in os.listdir(dir):
            if f.endswith('.xml') and not f.endswith('.img.xml') and not f.endswith('.aux.xml'):
                root = ET.parse(os.path.join(dir, f)).getroot()

                coord = [ float(root.find(f'idinfo/spdom/bounding/{c}').text) for c in coord_list ]
                if (coord[0] - target[0]) * (coord[1] - target[0]) < 0 and (coord[2] - target[1]) * (coord[3] - target[1]) < 0:
                    print(target, f)

def read_tif():
    def remove_outliers(data, sigma=8):
        std = np.std(data)
        mean = np.mean(data)
        data = np.array([ d if abs(d - mean) < sigma * std else mean for d in data ])
        return data
    
    file_path = 'a5_las_lasda_23.tif'

    dsm = rxr.open_rasterio(file_path, masked=True).squeeze()
    dsm_array = np.array(dsm)

    # check distribution of height information
    fig = plt.figure()
    plt.plot(dsm_array.flatten())
    plt.savefig('outlier.png')

    # remove outliers
    dsm_array = remove_outliers(dsm_array.ravel()).reshape(dsm_array.shape)

    # resize to the image size of the RGB image
    json_file = 'instances_default.json'
    img = COCO(json_file).imgs[1]
    dsm_array = resize(dsm_array, (img['height'], img['width']))
    np.save('dem.npy', dsm_array.astype(float))
    
    # export the plot
    fig, ax = plt.subplots(dpi=500)
    im = ax.imshow(dsm_array, cmap='viridis', vmin=dsm_array.min(), vmax=dsm_array.max())
    v = np.linspace(dsm_array.min(), dsm_array.max(), 15, endpoint=True)
    fig.colorbar(im, ticks=v)
    plt.savefig('dem.png')

def append_to_las():
    las_list = ['stratmap18-50cm_2995222a3output.las']
    las_out = "stratmap18-50cm_2995222a1output.las"

    for las in las_list:
        with laspy.open(las_out, mode='a') as outlas:
            with laspy.open(las) as inlas:
                for points in inlas.chunk_iterator(2_000_000):
                    outlas.append_points(points)

def print_npy():
    filename = 'label.npy'

    label = np.load(filename)
    df = pd.DataFrame(label)
    df.to_csv('data.csv', index=False)

    print(label)

def mask_filter():
    # json_file = 'combine.json'
    json_file = 'instances_default.json'

    coco = COCO(json_file)

    img = coco.imgs[1]
    blank_mask = np.zeros((img['height'], img['width']))

    # background: 0, asphalt: 1, building: 2, grass: 4, pedestrian walk: 6, tree, 8 
    for cat_id in [4, 2, 8, 6]:
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_id, iscrowd=None)
        anns = coco.loadAnns(anns_ids)

        if len(anns) != 0:
            mask = coco.annToMask(anns[0])
            for i in range(1, len(anns)):
                mask += coco.annToMask(anns[i])

            mask_mask = np.invert(np.logical_and(mask, blank_mask))
            blank_mask = (mask >= 1) * cat_id + np.multiply(blank_mask, mask_mask)
            
            plt.imsave(f'output/{cat_id}.png', blank_mask)
            plt.imsave(f'output/{cat_id}_mask.png', mask)

    np.save('label.npy', blank_mask.astype(int))
    lookup = np.array([1, 1, 2, 0, 5, 6, 100, 7, 101])
    blank_mask = lookup[blank_mask.astype(int)] 

    # asphalt: 1, building: 2, grass: 5, pedestrian walk: 100, tree, 101
    print_npy()

def shift():
    code = 'C6'

    json_file = f'data/{code}.json'
    data = json.load(open(json_file, "r"))

    # coco = COCO(json_file)
    # img = coco.imgs[1]
    # anns_ids = coco.getAnnIds(imgIds=img['id'])
    # anns = coco.loadAnns(anns_ids)
    shift_list = [6, 5]
    width, height = data['images'][0]['width'] - shift_list[0] * 2, data['images'][0]['height'] - shift_list[1] * 2

    # shift
    for a in data['annotations']:
        a['segmentation'][0] = [coord - shift_list[i % 2]  if coord - shift_list[i % 2] >= 0 else 0 for i, coord in enumerate(a['segmentation'][0])]
        a['segmentation'][0] = [width if coord > width and not i % 2 else coord for i, coord in enumerate(a['segmentation'][0])]
        a['segmentation'][0] = [height if coord > height and i % 2 else coord for i, coord in enumerate(a['segmentation'][0])]
    
    # copy for RGB image
    anntation_length = len(data['annotations'])
    for a in data['annotations']:
        if anntation_length >= a['id']:
            new_a = a.copy()
            new_a['id'] = anntation_length + new_a['id']
            new_a['image_id'] = 2
            data['annotations'].append(new_a)
        
    with open("test.json", "w") as f:
        json.dump(data, f)

def resize_annotation():
    code = 'C6'
    
    json_file = f'data/{code}.json'
    # json_file = 'test.json'
    data = json.load(open(json_file, "r"))
    w1, h1 = data['images'][0]['width'], data['images'][0]['height']
    
    png_file = f'data/{code}.png'
    image = Image.open(png_file)
    w2, h2 = image.size
    w2, h2 = 2388, 1727

    ratio_w = w2 / w1
    ratio_h = h2 / h1
    ratio = [ratio_w, ratio_h]

    # shift
    for a in data['annotations']:
        a['segmentation'][0] = [coord * ratio[i % 2] for i, coord in enumerate(a['segmentation'][0])]
        a['segmentation'][0] = [w2 if coord > w2 and not i % 2 else coord for i, coord in enumerate(a['segmentation'][0])]
        a['segmentation'][0] = [h2 if coord > h2 and i % 2 else coord for i, coord in enumerate(a['segmentation'][0])]
    
        
    with open("test.json", "w") as f:
        json.dump(data, f)

if __name__ == '__main__':
    # main()
    # read_img()
    # find_file()
    # read_tif()

    # append_to_las()
    # print_npy()
    # mask_filter()

    # shift()
    resize_annotation()