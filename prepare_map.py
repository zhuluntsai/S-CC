import matplotlib.pyplot as plt
import rioxarray as rxr
import numpy as np
from PIL import Image
from skimage.transform import resize

def export_raster(code):
    def remove_outliers(data, sigma=4):
        std = np.std(data)
        mean = np.mean(data)
        data = np.array([ d if abs(d - mean) < sigma * std else mean for d in data ])
        return data
    
    tif_file = f'data/{code}.tif'
    # tif_file = f'data/a5_las_lasda_23.tif'
    png_file = f'data/{code}.png'
    output_path = f'output/{code}_dem.png'
    
    dsm = rxr.open_rasterio(tif_file, masked=True).squeeze()
    dsm_array = np.array(dsm)

    image = Image.open(png_file)
    w, h = image.size

    dsm_array = resize(dsm_array, (h, w))
    dsm_array = remove_outliers(dsm_array.ravel()).reshape(dsm_array.shape)
    plt.imsave(output_path, dsm_array, cmap='gray', vmin=dsm_array.min(), vmax=dsm_array.max() )
    print(f'{output_path} are saved')

if __name__ == '__main__':
    code = 'C5'
    export_raster(code)