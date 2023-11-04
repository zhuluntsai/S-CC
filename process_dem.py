import json
import numpy as np
import rioxarray as rxr
from pycocotools.coco import COCO
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm
from noise_reduct import IQR, check_mask, box_filter

label_dict = {
    # -1: 'all',
    # 0: 'background',
    2: 'buildings',
    4: 'grass',
    6: 'pedestrian walk',
    8: 'trees', 
}

def combine_image_id(code):
    json_file = f'data/{code}.json'
    data = json.load(open(json_file, "r"))
    width, height = data['images'][0]['width'], data['images'][0]['height']
    output_path = f"output/json/{code}_combine.json"
    new_annotations = []

    for a in data['annotations']:
        if a['image_id'] == 1:
            if a['category_id'] in [4, 6]:
                new_annotations.append(a)
        elif a['image_id'] == 2:
            if a['category_id'] in [2, 8]:
                a['image_id'] = 1
                new_annotations.append(a)

    data['annotations'] = new_annotations
        
    with open(output_path, "w") as f:
        json.dump(data, f)
        print(f'{output_path} are saved')
        
def convert_to_mask(code):
    json_file = f'output/json/{code}_combine.json'
    coco = COCO(json_file)
    img = coco.imgs[1]
    blank_mask = np.zeros((img['height'], img['width']))

    # background: 0, asphalt: 1, building: 2, grass: 4, pedestrian walk: 6, tree, 8 
    for cat_id in [4, 2, 8, 6]:
        print(cat_id)
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
            print(f'output/{cat_id}.png are saved')

    np.save(f'output/label/{code}.npy', blank_mask.astype(int))
    return blank_mask.astype(int)

def mask_filter(code):
    tif_file = f'data/{code}.tif'
    dsm = rxr.open_rasterio(tif_file, masked=True).squeeze()
    dsm_array = np.array(dsm)

    label = np.load(f'output/label/{code}.npy')
    elevation = resize(dsm_array, label.shape)

    json_file = f'output/json/{code}_combine.json'
    coco = COCO(json_file)
    img = coco.imgs[1]
    sigma = 1

    for k in tqdm(list(label_dict.keys())):
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=k, iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        blank_mask = np.zeros((img['height'], img['width']))
        
        if k in [2, 8]:
            mask = coco.annToMask(anns[0])
            for i, annotation in enumerate(tqdm(anns)):
                mask = coco.annToMask(annotation)
                mask_elevation = elevation * mask
                mask_elevation = box_filter(mask_elevation, label, sigma, k) * mask
                blank_mask += mask_elevation

        if k in [4, 6]:
            blank_mask = elevation * (label == k)
            blank_mask = IQR(blank_mask.ravel()).reshape(label.shape)
            blank_mask = IQR(blank_mask.ravel()).reshape(label.shape)
            mask_elevation = box_filter(blank_mask, label, sigma, k) * (label == k)
        
        check_mask(mask_elevation, f'output/outlier/{code}_{k}_final.png')
    
    return mask_elevation

def stick():
    code_list = ['C5', 'C6']
    letters = sorted(set([ c[0] for c in code_list]))
    index = sorted(set([ c[1:] for c in code_list]))
    print(letters)
    print(index)

    json_file = f'output/json/{code_list[0]}_combine.json'
    coco = COCO(json_file)
    img = coco.imgs[1]
    h, w= img['height'], img['width']
    blank_mask = np.zeros((h * len(index), w * len(letters)))
    print(blank_mask.shape)
    
    for x, l in enumerate(letters):
        for y, i in enumerate(index):
            code = f'{l}{i}'
            if code in code_list:
                # tif_file = f'data/{code}.tif'
                # dsm = rxr.open_rasterio(tif_file, masked=True).squeeze()
                # dsm_array = np.array(dsm)
                # dsm_array = resize(dsm_array, (h, w))

                # label = np.load(f'output/label/{code}.npy')
                # print(label.shape)
                # print(h*y, h*(y+1), w*x, w*(x+1))
                # blank_mask[h*y: h*(y+1), w*x: w*(x+1)] = label

                mask_elevation = mask_filter(code)
                blank_mask[h*y: h*(y+1), w*x: w*(x+1)] = mask_elevation

    plt.imsave(f'output/test.png', blank_mask)
    # np.savetxt("test.csv", blank_mask[1600:1700, :].astype(int), delimiter=",")



if __name__ == '__main__':
    code = 'D7'

    combine_image_id(code)
    convert_to_mask(code)

    # mask_filter(code)
    # stick()