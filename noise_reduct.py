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

label_dict = {
    -1: 'all',
    0: 'background',
    2: 'buildings',
    4: 'grass',
    6: 'pedestrian walk',
    8: 'trees', 
}

def remove_outliers(data, sigma=8):
    std = np.std(data)
    mean = np.mean(data)
    data = np.array([ d if abs(d - mean) < sigma * std else mean for d in data ])
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

def main():
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
        if point[3] in [0, 4, 6]:
            if point[2] < lowerbound:
                keep_point.append(point)
            else:
                remove_point.append(point)
        elif point[3] in [2, 8]:
            if point[2] > lowerbound and point[2] < upperbound:
                keep_point.append(point)
            else:
                remove_point.append(point)
    keep_point = np.array(keep_point)
    remove_point = np.array(remove_point)

    # for label in list(label_dict.keys()):
    #     check_outlier(new_point, label)
    check_z(keep_point, remove_point)

    # geom = open3d.geometry.PointCloud()
    # geom.points = open3d.utility.Vector3dVector(new_point[:, :3])
    # cmap = mpl.colors.ListedColormap(["gray", "black", "yellow", "black", "lime", "black", "red", "blue", "green"])
    # geom.colors = open3d.utility.Vector3dVector(cmap(new_point[:, 3].astype(int))[:, :3])
    # # open3d.visualization.draw_geometries([geom])
    # open3d.io.write_point_cloud('label.pcd', geom)



if __name__ == '__main__':
    main()