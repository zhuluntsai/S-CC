import laspy
import open3d
import os
import pandas as pd
import numpy as np
import rasterio
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
from pycocotools.coco import COCO
import matplotlib.cm as cm
import copy
import rioxarray as rxr
from skimage.transform import resize
import osmnx as ox
import geopandas as gpd
from pyproj import Transformer
from shapely import Polygon
import shapefile
from rasterio.mask import mask
import geopandas as gpd
import regionmask
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

label_dict = {
    # -1: 'all',
    # 0: 'background',
    # 2: 'buildings',
    4: 'grass',
    6: 'pedestrian walk',
    # 8: 'trees', 
}

def IQR(data):
    temp = [d for d in data if d != 0]
    q75, q25 = np.percentile(temp, [75, 25])
    iqr = q75 - q25
    iqr_span = 0
    upper_bound = q75 + iqr_span * iqr
    lower_bound = q25 - iqr_span * iqr
    median = np.median(temp)
    data = np.array([ d if d > lower_bound and d < upper_bound else 0 for d in data ])
    # data = np.array([ d for d in data if d > q25 and d < q75 ])
    return data

def standard_deviation(data, sigma):
    temp = [d for d in data if d != 0]
    std = np.std(temp)
    mean = np.mean(temp)
    print(std, mean)
    data = np.array([ d if abs(d - mean) < sigma * std else 0 for d in data ])
    # data = np.array([ d for d in data if abs(d - mean) < sigma * std ])
    return data

def check_outlier(data, label, ann_id):
    label_name = label_dict[label]
    binwidth = 0.1
    sigma = 1
    data = [d for d in data.ravel() if d != 0]
    bins = np.arange(np.min(data), np.max(data) + binwidth, binwidth)

    fig = plt.figure(dpi=500)
    fig.suptitle(f'label: {label_name}, point amount: {len(data)}')
    plt.subplots_adjust(hspace=1.2)

    ax1 = plt.subplot(211)
    ax1.title.set_text(f'original')
    y, x, _ = plt.hist(data, bins=bins)
    # print(y, x)
    plt.xlabel('z (height)')

    for i in range(1):
        data = standard_deviation(data, sigma)
        count = len([ d for d in data if d == 0 ])
        print(i, count)
    # data = standard_deviation(data, sigma)
    ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
    ax2.title.set_text(f'mask, sigma:{sigma}')
    plt.hist(data, bins=bins)
    plt.xlabel('z (height)')

    plt.savefig(f'output/outlier/{label}_{ann_id}.png')
    print(f'output/outlier/{label}_{i}.png are saved')

def check_z(point_data, remove_point):
    label_list = list(label_dict.keys())
    xlim_min = 5
    xlim_max = 35
    bins = np.linspace(xlim_min, xlim_max, (xlim_max - xlim_min) * 10)
    fig = plt.figure(dpi=500)
    plt.subplots_adjust(hspace=1, wspace=0.3)
    ax = []
    point_list = [[] for x in range(len(label_list))]
    point_list[0] = [point[2] for point in point_data]
    remove_point_list = [[] for x in range(len(label_list))]
    # remove_point_list[0] = [point[2] for point in remove_point]
    for point in point_data:
        point_list[label_list.index(point[3])].append(point[2])
    for point in remove_point:
        remove_point_list[label_list.index(point[3])].append(point[2])
        
    for i, (k, v) in enumerate(label_dict.items()):
        height = point_list[i]
        remove = remove_point_list[i]
        ax.append(plt.subplot(int(f'32{i + 1}')))
        ax[-1].title.set_text(f'{v} ({len(height)})')
        print(v, len(height), len(remove))
        plt.hist(remove, bins=bins)
        plt.hist(height, bins=bins)
        plt.xlim(xlim_min, xlim_max)
        # plt.xlabel('z (height)')
    
    plt.savefig(f'output/z.png')
    print(f'output/z.png are saved')

def check_mask(data, filename):
    fig = plt.figure(dpi=500)

    ax = plt.subplot(211)
    binwidth = 0.1
    temp = [ d for d in data.ravel() if d != 0 ]
    bins = np.arange(np.min(temp), np.max(temp) + binwidth, binwidth)
    plt.hist(temp, bins=bins)

    ax = plt.subplot(212)
    im = ax.imshow(data, cmap='viridis', vmin=data.min(), vmax=data.max())
    v = np.linspace(data.min(), data.max(), 15, endpoint=True)
    fig.colorbar(im, ticks=v)
    plt.savefig(filename)
    print(f'{filename} are saved')

def check_boxplot(elevation, label):
    box_list = [ [ e for e, l in zip(elevation.ravel(), label.ravel()) if l == k] for k in list(label_dict.keys()) ]
    
    fig = plt.figure(dpi=500)
    ax1 = plt.subplot(111)
    plt.boxplot(box_list, labels=list(label_dict.values()))
    plt.savefig(f'output/outlier_boxplot.png')

    elevation = IQR(elevation.ravel(), 1).reshape(elevation.shape)
    fig, ax = plt.subplots(dpi=500)
    im = ax.imshow(elevation, cmap='viridis', vmin=elevation.min(), vmax=elevation.max())
    v = np.linspace(elevation.min(), elevation.max(), 15, endpoint=True)
    fig.colorbar(im, ticks=v)
    plt.savefig(f'output/test.png')

def box_filter(elevation, label, sigma, cat_id):
    height, width = elevation.shape
    temp = [d for d in elevation.ravel() if d != 0]
    std = np.std(temp)
    mean = np.mean(temp)

    kernel = np.ones((5, 5))
    kernel_center = [int(kernel.shape[0]/2), int(kernel.shape[1]/2)]
    count = [0, 0]

    new_image = np.zeros(elevation.shape)
    for x in range(0, width, 1):
        if np.sum((elevation[:, x] > 0)) == 0:
            continue
        for y in range(0, height, 1):
            pixel = elevation[y, x]

            if pixel == 0 and label[y, x] == cat_id:
                new_image[y, x] = mean
                continue
            elif mean - std * sigma < pixel and pixel < mean + std * sigma:
                new_image[y, x] = pixel
                count[0] += 1
                continue
            elif pixel == 0:
                continue

            kernel_list = []
            for i in range(0, kernel.shape[0], 1):
                for j in range(0, kernel.shape[1], 1):
                    if kernel[i, j] == 1:
                        put_pixel_x = x + i - kernel_center[0]
                        put_pixel_y = y + j - kernel_center[1]
                        try:
                            kernel_list.append(elevation[put_pixel_y, put_pixel_x])
                        except:
                            pass
            
            kernel_list = [k for k in kernel_list if k != 0]
            if len(kernel_list) == 0:
                kernel_list.append(0)
            new_image[y, x] = np.mean(kernel_list)
    
    return new_image

def mask_filter(elevation, label):
    json_file = 'combine.json'

    coco = COCO(json_file)

    img = coco.imgs[1]
    sigma = 1

    for k in list(label_dict.keys()):
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=k, iscrowd=None)
        anns = coco.loadAnns(anns_ids)

        max_area = 0
        max_anns = anns[0]
        for annotation in anns:
            area = annotation['area']
            if area > max_area:
                max_area = area
                max_anns = annotation

        blank_mask = np.zeros((img['height'], img['width']))
        blank_mask = elevation * (label == k)
        
        # mask = coco.annToMask(anns[0])
        # for i, annotation in enumerate(tqdm(anns)):
        #     mask = coco.annToMask(annotation)
        #     mask_elevation = elevation * mask
            # mask_elevation = box_filter(mask_elevation, label, sigma, k) * mask

            # blank_mask += mask_elevation

            # check_mask(mask_elevation, f'output/{k}_3.png')
            # check_outlier(mask_elevation, k, i)
        
        # blank_mask = standard_deviation(blank_mask.ravel(), sigma).reshape(blank_mask.shape)
        # blank_mask = standard_deviation(blank_mask.ravel(), sigma).reshape(blank_mask.shape)
        blank_mask = IQR(blank_mask.ravel()).reshape(label.shape)
        blank_mask = IQR(blank_mask.ravel()).reshape(label.shape)
        check_mask(blank_mask, f'output/outlier/{k}_before.png')
        mask_elevation = box_filter(blank_mask, label, sigma, k) * (label == k)
        
        check_mask(mask_elevation, f'output/outlier/{k}_after.png')


    # mask_mask = np.invert(np.logical_and(mask, blank_mask))
    # blank_mask = (mask >= 1) * k + np.multiply(blank_mask, mask_mask)
        
    # blank_mask = blank_mask * elevation
    # mask_mask = np.invert(mask_mask) * 1
    # fig, ax = plt.subplots(dpi=500)
    # im = ax.imshow(blank_mask, cmap='viridis', vmin=blank_mask.min(), vmax=blank_mask.max())
    # cmap = copy.copy(mpl.cm.get_cmap("rainbow"))
    # cmap.set_under('w', alpha=0)
    # ax.imshow(mask_mask * mask_mask, cmap=cmap, alpha=1, vmin=mask_mask.min() + 0.001)
    # plt.savefig('output/overlap2.png')
         
def check_dem_label(blank_elevation, label):
    # blank_elevation = np.multiply(blank_elevation, label)
    blank_elevation = standard_deviation(blank_elevation.ravel(), 9).reshape(blank_elevation.shape)
    # fig, ax = plt.subplots(dpi=500)
    # im = ax.imshow(blank_elevation, cmap='viridis', vmin=blank_elevation.min(), vmax=blank_elevation.max())
    # v = np.linspace(blank_elevation.min(), blank_elevation.max(), 15, endpoint=True)
    # fig.colorbar(im, ticks=v)
    # plt.savefig('output/dem_from_point_cloud.png')

    # kernel = np.ones((3, 3), np.uint8)
    # blank_elevation = cv2.blur(blank_elevation, (3, 3))
    # blank_elevation = cv2.dilate(blank_elevation, (3, 3))
    # blank_elevation = cv2.dilate(blank_elevation, kernel)
    blank_elevation = boxfilter(blank_elevation, label)

    cmap = mpl.colors.ListedColormap(["gray", "black", "orange", "black", "lime", "black", "red", "blue", "green"])
    # ax3 = plt.subplot(313)
    fig = plt.figure(dpi=500)
    section = [650, 580]
    print(blank_elevation.shape)
    x = blank_elevation[section[1], :]
    x_label = label[section[1], :]
    y = blank_elevation[:, section[0]]
    y_label = label[:, section[0]]

    # x = batch(x)
    # y = batch(y)

    gs = fig.add_gridspec(2, 2,  width_ratios=(12, 1), height_ratios=(1, 8),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(blank_elevation)
    # ax.axvline(x=section[0], color='r')
    # ax.axhline(y=section[1], color='r')
    ax.scatter(np.repeat(section[0], len(y)), np.arange(len(y)), color=cmap(y_label)[:, :3], s=0.1)
    ax.scatter(np.arange(len(x)), np.repeat(section[1], len(x)), color=cmap(x_label)[:, :3], s=0.1)

    ax_x = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_y = fig.add_subplot(gs[1, 1], sharey=ax)

    ax_x.tick_params(axis="x", labelbottom=False)
    ax_y.tick_params(axis="y", labelleft=False)


    ax_x.plot(x, alpha=0.2)
    ax_x.scatter(np.arange(len(x)), x, color=cmap(x_label)[:, :3], s=0.2)
    ax_y.plot(y, np.arange(len(y)), alpha=0.2)
    ax_y.scatter(y, np.arange(len(y)), color=cmap(y_label)[:, :3], s=0.2)
    plt.savefig('output/dem_from_point_cloud.png')

def las_to_elevation():
    las = laspy.read("stratmap18-50cm_2995222a1output(2).las")
    point_data_1 = np.stack([las.x, las.y, las.z, las.classification], axis=0).transpose((1, 0))

    las = laspy.read("stratmap18-50cm_2995222a3output(2).las")
    point_data_2 = np.stack([las.x, las.y, las.z, las.classification], axis=0).transpose((1, 0))

    point_data = np.vstack((point_data_1, point_data_2))

    point_min = np.min(point_data, axis=0)
    delta =  np.max(point_data, axis=0) - np.min(point_data, axis=0)
    print(delta * 2)
    print(point_data.shape)

    label = np.load('label.npy')
    print(label.shape)

    lowerbound = 10.3
    upperbound = 35

    keep_point = []
    remove_point = []

    for i, point in enumerate(tqdm(point_data)):
        coord = ((point - point_min) * 2).astype(int)
        point[3] = int(label[label.shape[0] - coord[1] - 1, coord[0]])
        # if point[3] in [0, 4, 6]:
        #     if point[2] < lowerbound:
        #         keep_point.append(point)
        #     else:
        #         remove_point.append(point)
        # elif point[3] in [2, 8]:
        #     if point[2] > lowerbound and point[2] < upperbound:
        #         keep_point.append(point)
        #     else:
        #         remove_point.append(point)
    
    keep_point = np.array(keep_point)
    remove_point = np.array(remove_point)

    elevation = np.zeros(label.shape)
    for point in tqdm(point_data):
        # print(point)
        coord = ((point - point_min) * 2).astype(int)
        elevation[label.shape[0] - coord[1] - 1, coord[0]] = point[2]
    np.save('elevation.npy', elevation.astype(float))

def point_visualization(point):
    geom = open3d.geometry.PointCloud()
    geom.points = open3d.utility.Vector3dVector(point[:, :3])
    cmap = mpl.colors.ListedColormap(["gray", "black", "yellow", "black", "lime", "black", "red", "blue", "green"])
    geom.colors = open3d.utility.Vector3dVector(cmap(point[:, 3].astype(int))[:, :3])
    
    # open3d.visualization.draw_geometries([geom])
    open3d.io.write_point_cloud('label.pcd', geom)

# def main():
    # elevation = np.load('elevation.npy')
    # label = np.load('label.npy')

    # file_path = 'data/a5_las_lasda_23.tif'
    # dsm = rxr.open_rasterio(file_path, masked=True).squeeze()
    # dsm_array = np.array(dsm)
    # elevation = resize(dsm_array, elevation.shape)

    # # check_dem_label(elevation, label)
    # # check_boxplot(elevation, label)
    # mask_filter(elevation, label)

# if __name__ == '__main__':
#     main()