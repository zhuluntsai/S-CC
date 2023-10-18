import pandas as pd
from tqdm import tqdm 
import os
import matplotlib.pyplot as plt
import string
import numpy as np
import csv
from pyproj import Transformer
from tqdm import tqdm

def export_csv():
    directory = 'Formatted Data'

    df_list = []
    speed_list = []
    for root, dirs, files in tqdm(os.walk(directory)):
        for filename in files:
            path = os.path.join(root, filename)
            if 'GPS' in path:
                df = pd.read_excel(path)
                p = path.split('/')
                # df['subject'] = [ p[1] for _ in range(len(df)) ]
                # df['date'] = [ p[3] for _ in range(len(df)) ]
                # df['trip'] = [ p[4] for _ in range(len(df)) ]

                max_speed = df['speed'].max()
                if max_speed < 2.5:
                    speed_list.append(max_speed)
                    
                    df = df.drop(columns=['timestamp', 'speed', 'accuracy', 'bearing', 'altitude'])
                    df_list.append(df)

    plt.hist(speed_list, bins=30)
    plt.xlabel('speed')
    plt.ylabel('frequency')
    plt.savefig('test.png')
    # print(speed_list)
    print(len(df_list))
    pd.concat(df_list).to_csv('test.csv', index=False)

def grid_search(array, target):
    for i, a in enumerate(array):
        temp = a - target
        if temp > 0:
            return i
    return -1

def plot_frequency(frequency_dict):
    # sort dict by value
    keys = list(frequency_dict.keys())
    values = list(frequency_dict.values())
    sorted_value_index = np.flip(np.argsort(values))
    sorted_dict = {keys[i]: values[i] for i in sorted_value_index if values[i] != 0}
    
    # plot
    fig = plt.figure(dpi=500, figsize=(20,6))
    x = np.arange(0, len(sorted_dict), 1)
    y = list(sorted_dict.values())
    label = list(sorted_dict.keys())

    # add info
    plt.bar(x, y)
    for i in range(len(x)):
        plt.text(i, y[i], int(y[i]), ha='center')
    plt.xticks(np.arange(0, len(sorted_dict), 1), labels=label)
    plt.tight_layout()
    plt.savefig('frequency.png')

def plot_scatter(x_grid, y_grid, points, letters, index):
    fig = plt.figure(dpi=500)
    x = np.tile(np.arange(1, len(x_grid), 1), len(y_grid) - 1)
    y = np.repeat(np.arange(len(y_grid) - 1, 0, -1), len(x_grid) - 1)
    s = points.ravel() * 0.05

    plt.xticks(np.arange(1, len(x_grid), 1), labels=letters)
    plt.yticks(np.arange(1, len(y_grid), 1), labels=np.flip(index))
    plt.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.scatter(x, y, s=s, alpha=0.5)
    plt.savefig('frequency.png')

def extract_frequency():
    letters = list(string.ascii_uppercase)[:12]
    index = np.arange(1, 14, 1)
    print(letters)
    print(index)

    x_start = 275948
    width = 594

    y_start = 3288502
    height = 420
    
    x_grid = [x_start, ]
    y_grid = [y_start, ]
    
    for x, l in enumerate(letters):
        x_max = x_start + width * (x + 1)
        x_grid.append(x_max)

    for y, i in enumerate(index):
        y_max = y_start + height * (y + 1)
        y_grid.append(y_max)

    points = [ [ [] for _ in range(len(x_grid)) ] for _ in range(len(y_grid)) ]
    points = np.zeros((len(y_grid) - 1, len(x_grid) - 1))

    transformer = Transformer.from_crs( "EPSG:4326", "EPSG:6344")
    with open('test.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        next(reader)
        for row in tqdm(reader):
            latitude, longitude = map(float, row[0].split(','))
            latitude, longitude = transformer.transform(latitude, longitude)

            x = grid_search(x_grid, latitude)
            y = len(y_grid) - grid_search(y_grid, longitude)
            if x > 0 and x < len(y_grid) and y > -1 and y < len(x_grid):
                points[y - 1, x - 1] += 1

        print(points.astype(int))
        print(np.sum(points))

    # Create frequency dict
    label = [ f'{l}{i}' for i in index for l in letters ]
    frequency_dict = {}
    for p, l in zip(points.ravel(), label):
        frequency_dict[l] = p

    plot_scatter(x_grid, y_grid, points, letters, index) 
    plot_frequency(frequency_dict)

def main():
    # export_csv()
    extract_frequency()

if __name__ == '__main__':
    main()