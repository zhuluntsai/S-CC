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

label_dict = {
    -1: 'all',
    0: 'background',
    2: 'buildings',
    4: 'grass',
    6: 'pedestrian walk',
    8: 'trees', 
}

def IQR(data, iteration):
    q75, q25 = np.percentile(data, [75, 25])
    median = np.median(data)
    # data = np.array([ d if d > q25 and d < q75 else median for d in data ])
    data = np.array([ d for d in data if d > q25 and d < q75 ])
    return data

def standard_deviation(data, sigma):
    temp = [d for d in data if d != 0]
    std = np.std(temp)
    mean = np.mean(temp)
    print(std, mean)
    data = np.array([ d if abs(d - mean) < sigma * std else 0 for d in data ])
    # data = np.array([ d for d in data if abs(d - mean) < sigma * std ])
    return data

def check_outlier(point_data, label):
    label_name = label_dict[label]

    xlim_min = 5
    xlim_max = 35
    bins = np.linspace(xlim_min, xlim_max, (xlim_max - xlim_min) * 30)
    
    xlim_zoom_min = 9
    xlim_zoom_max = 12

    lowerbound = 0
    upperbound = 10.3
    if label in [2, 8]: 
        lowerbound = upperbound
        upperbound = 35
    xlim_remove_min = max(lowerbound, xlim_min)
    xlim_remove_max = min(upperbound, xlim_max)
    bins_remove = np.linspace(xlim_remove_min, xlim_remove_max, int(xlim_remove_max - xlim_remove_min) * 30)

    height = [ point[2] for point in point_data if point[3] == label]
    fig = plt.figure(dpi=500)
    fig.suptitle(f'label: {label_name}, point amount: {len(height)}')
    plt.subplots_adjust(hspace=1.2)

    ax1 = plt.subplot(311)
    ax1.title.set_text('original')
    plt.hist(height, bins=bins)
    plt.axvspan(xlim_zoom_min, xlim_zoom_max, color='orange', alpha=0.5)
    plt.xlim(xlim_min, xlim_max)
    plt.xlabel('z (height)')
    
    ax2 = plt.subplot(312)
    ax2.title.set_text('zoom in')
    plt.hist(height, bins=bins, color='orange')
    plt.xlim(xlim_zoom_min, xlim_zoom_max)
    plt.xticks(np.arange(xlim_zoom_min, xlim_zoom_max, step=0.2), rotation=45)
    plt.xlabel('z (height)')

    ax3 = plt.subplot(313)
    ax3.title.set_text('remove outliers')
    height_remove =  [ point[2] for point in point_data if point[3] == label and point[2] > lowerbound and point[2] < upperbound]
    plt.hist(height, bins=bins)
    plt.hist(height_remove, bins=bins_remove, color='orange')
    plt.xlim(xlim_min, xlim_max)
    plt.xlabel('z (height)')

    plt.savefig(f'output/outlier_{label}.png')
    print(f'output/outlier_{label}.png are saved')

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

def check_mask(dsm_array, filename):
    fig, ax = plt.subplots(dpi=500)
    im = ax.imshow(dsm_array, cmap='viridis', vmin=dsm_array.min(), vmax=dsm_array.max())
    v = np.linspace(dsm_array.min(), dsm_array.max(), 15, endpoint=True)
    fig.colorbar(im, ticks=v)
    plt.savefig(filename)


def batch(x_list):
    for i, x in enumerate(x_list):
        try:
            if x == 0:
                x_list[i] = (x_list[i - 1] + x_list[i + 1]) / 2
        except:
            pass
    return x_list

def boxfilter(elevation, label):
    height, width = elevation.shape

    kernel = np.ones((7, 7))
    kernel_center = [int(kernel.shape[0]/2), int(kernel.shape[1]/2)]

    new_image = np.zeros(elevation.shape)
    for x in range(0, width, 1):
        if np.mean(elevation[:, x]) == 0:
            continue
        for y in range(0, height, 1):
            pixel = elevation[y, x]

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
    json_file = 'instances_default.json'

    coco = COCO(json_file)

    img = coco.imgs[1]
    sigma = 5

    # anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=2, iscrowd=None)
    anns_ids = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(anns_ids)

    # max_area = 0
    # max_anns = anns[0]
    # for annotation in anns:
    #     area = annotation['area']
    #     if area > max_area:
    #         max_area = area
    #         max_anns = annotation

    blank_mask = np.zeros((img['height'], img['width']))
    for annotation in tqdm(anns):
        mask = coco.annToMask(annotation)
        mask_elevation = elevation * mask
        blank_mask += boxfilter(mask_elevation, label) * mask
        # blank_mask += mask_elevation
        # mask_elevation = standard_deviation(mask_elevation.ravel(), sigma).reshape(mask_elevation.shape)
        
    check_mask(blank_mask, 'output/test_all.png')

    # label_name = label_dict[2]
    # binwidth = 0.1
    
    # height = [h for h in elevation.flatten() if h != 0]
    # bins = np.arange(np.min(height), np.max(height) + binwidth, binwidth)
    # print(bins)

    # fig = plt.figure(dpi=500)
    # fig.suptitle(f'label: {label_name}, point amount: {len(height)}')
    # plt.subplots_adjust(hspace=1.2)

    # mask = coco.annToMask(max_anns)
    # ax1 = plt.subplot(211)
    # ax1.title.set_text(f'original')
    # plt.hist(height, bins=bins)
    # plt.xlabel('z (height)')

    # mask_elevation = [h for h in mask_elevation.flatten() if h != 0]
    # ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
    # ax2.title.set_text(f'mask, sigma:{sigma}')
    # plt.hist(mask_elevation, bins=bins)
    # plt.xlabel('z (height)')

    # plt.savefig(f'output/outlier_z.png')
    # print(f'output/outlier_z.png are saved')
        

def plot_dem_label(blank_elevation, label):
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


def main():
    # las = laspy.read("stratmap18-50cm_2995222a1output(2).las")
    # point_data_1 = np.stack([las.x, las.y, las.z, las.classification], axis=0).transpose((1, 0))

    # las = laspy.read("stratmap18-50cm_2995222a3output(2).las")
    # point_data_2 = np.stack([las.x, las.y, las.z, las.classification], axis=0).transpose((1, 0))

    # point_data = np.vstack((point_data_1, point_data_2))

    # point_min = np.min(point_data, axis=0)
    # delta =  np.max(point_data, axis=0) - np.min(point_data, axis=0)
    # print(delta * 2)
    # print(point_data.shape)

    # label = np.load('label.npy')
    # print(label.shape)

    # lowerbound = 10.3
    # upperbound = 35

    # keep_point = []
    # remove_point = []

    # for i, point in enumerate(tqdm(point_data)):
    #     coord = ((point - point_min) * 2).astype(int)
    #     point[3] = int(label[label.shape[0] - coord[1] - 1, coord[0]])
    #     # if point[3] in [0, 4, 6]:
    #     #     if point[2] < lowerbound:
    #     #         keep_point.append(point)
    #     #     else:
    #     #         remove_point.append(point)
    #     # elif point[3] in [2, 8]:
    #     #     if point[2] > lowerbound and point[2] < upperbound:
    #     #         keep_point.append(point)
    #     #     else:
    #     #         remove_point.append(point)
    # keep_point = np.array(keep_point)
    # remove_point = np.array(remove_point)

    # elevation = np.zeros(label.shape)
    # for point in tqdm(point_data):
    #     # print(point)
    #     coord = ((point - point_min) * 2).astype(int)
    #     elevation[label.shape[0] - coord[1] - 1, coord[0]] = point[2]
    # np.save('elevation.npy', elevation.astype(float))

    elevation = np.load('elevation.npy')
    label = np.load('label.npy')
    # plot_dem_label(elevation, label)

    mask_filter(elevation, label)
    exit()
    

    # for label in list(label_dict.keys()):
    #     check_outlier(new_point, label)
    # check_z(keep_point, remove_point)

    # geom = open3d.geometry.PointCloud()
    # geom.points = open3d.utility.Vector3dVector(new_point[:, :3])
    # cmap = mpl.colors.ListedColormap(["gray", "black", "yellow", "black", "lime", "black", "red", "blue", "green"])
    # geom.colors = open3d.utility.Vector3dVector(cmap(new_point[:, 3].astype(int))[:, :3])
    # # open3d.visualization.draw_geometries([geom])
    # open3d.io.write_point_cloud('label.pcd', geom)



if __name__ == '__main__':
    main()