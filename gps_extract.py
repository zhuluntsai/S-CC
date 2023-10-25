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
    directory = 'gps_data'

    df_list = []
    speed_list = []
    for root, dirs, files in tqdm(os.walk(directory)):
        for filename in files:
            path = os.path.join(root, filename)
            if 'GPS' in path:
                df = pd.read_excel(path)
                p = path.split('/')
                df['subject'] = [ p[1] for _ in range(len(df)) ]
                df['date'] = [ p[3] for _ in range(len(df)) ]
                df['trip'] = [ p[4] for _ in range(len(df)) ]

                max_speed = df['speed'].max()
                # if max_speed < 2.5:
                #     speed_list.append(max_speed)
                    
                #     df = df.drop(columns=['timestamp', 'speed', 'accuracy', 'bearing', 'altitude'])
                df_list.append(df)

    plt.hist(speed_list, bins=30)
    plt.xlabel('speed')
    plt.ylabel('frequency')
    plt.savefig('speed.png')
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
    fig = plt.figure(dpi=500, figsize=(10,6))
    x = np.arange(0, len(sorted_dict), 1)
    y = list(sorted_dict.values())
    label = list(sorted_dict.keys())

    # add info
    plt.bar(x, y)
    for i in range(len(x)):
        plt.text(i, y[i], int(y[i]), ha='center')
    plt.xticks(np.arange(0, len(sorted_dict), 1), labels=label)
    plt.xlabel('subregion')
    plt.ylabel('trip')
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
    trip_list = []
    unknown_trip_list = []
    row_list = []

    transformer = Transformer.from_crs( "EPSG:4326", "EPSG:6344")
    with open('test.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        row0 = next(reader)
        row0 = row0[0].split(',')
        row0.append('code')
        row0.append('trip_id')
        for row in reader:
            row = row[0].split(',')
            try:
                latitude, longitude, trip = row[1], row[2], row[9]
            except:
                continue
            latitude, longitude = transformer.transform(latitude, longitude)

            x = grid_search(x_grid, latitude)
            y = len(y_grid) - grid_search(y_grid, longitude)

            if trip not in trip_list:
                if 'trip' in trip:
                    trip_list.append(trip)
                else:
                    unknown_trip_list.append(trip)

            if x > 0 and x < len(y_grid) and y > -1 and y < len(x_grid) and trip in trip_list:
                points[y - 1, x - 1] += 1
                code = f'{letters[x - 1]}{index[y - 1]}'
                trip_id = trip_list.index(trip)

                if code in ['A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13',
                            'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 
                            'C8', 'C9', 'C10', 'C11', 'C12', 'C13',
                            'D9', 'D10', 'D11', 'D12', 'D13',
                            'E12', 'E13' ]:
                    continue

                row.append(code)
                row.append(trip_id)
                row_list.append(row)
                # continue

    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        writer.writerow(row0)
        writer.writerows(row_list)

    df = pd.read_csv('output.csv')
    sorted_df = df.sort_values(by='trip_id')
    sorted_df.to_csv('sorted_output.csv', index=False)

    print(points.astype(int))
    print(np.sum(points))
    print(len(unknown_trip_list))

    # Create frequency dict
    label = [ f'{l}{i}' for i in index for l in letters ]
    frequency_dict = {}
    # points = [ p for p in points.ravel() if p > 5 ]
    for p, l in zip(points.ravel(), label):
        frequency_dict[l] = p

    # plot_scatter(x_grid, y_grid, points, letters, index) 
    plot_frequency(frequency_dict)

# def exclude():
#     read_csv_path = 'GPS_Data_Filtered.csv'
#     read_csv = open(read_csv_path, 'r', newline='')
#     reader = csv.reader(read_csv, delimiter=' ', quotechar='|')

#     write_csv_path = 'output.csv'
#     write_csv = open(write_csv_path, 'w', newline='')
#     writer = csv.writer(write_csv, lineterminator='\n')

#     writer.writerow(next(reader)[0].split(','))
#     for row in reader:
#         code = row[0].split(',')[10]
#         if code not in ['A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13',
#                             'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 
#                             'C8', 'C9', 'C10', 'C11', 'C12', 'C13',
#                             'D9', 'D10', 'D11', 'D12', 'D13',
#                             'E12', 'E13' ]:
#             writer.writerow(row[0].split(','))

    


def main():
    # export_csv()
    # extract_frequency()
    exclude()


if __name__ == '__main__':
    main()