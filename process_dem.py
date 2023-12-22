import json
import numpy as np
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from PIL import Image
from noise_reduct import IQR, check_mask, box_filter, check_dem_label
from skimage.transform import resize

label_dict = {
    # -1: 'all',
    # 0: 'background',
    1: 'asphalt',
    2: 'buildings',
    # 3: 'cobble stone',
    4: 'grass',
    # 5: 'bare soil',
    6: 'pedestrian walk',
    # 7: 'water',
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
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_id, iscrowd=None)
        anns = coco.loadAnns(anns_ids)

        if len(anns) != 0:
            mask = coco.annToMask(anns[0])
            for i in range(1, len(anns)):
                if anns[i]['segmentation'] == []:
                    print(anns[i]['id'])
                    continue
                mask += coco.annToMask(anns[i])

            mask_mask = np.invert(np.logical_and(mask, blank_mask))
            blank_mask = (mask >= 1) * cat_id + np.multiply(blank_mask, mask_mask)
    
            # plt.imsave(f'output/label/{cat_id}.png', blank_mask)
            # plt.imsave(f'output/label/{cat_id}_mask.png', mask)

    elevation = np.load(f'output/dem/{code}.npy')
    blank_mask = (blank_mask == 0) + blank_mask
    blank_mask = blank_mask * (elevation > 0)
    
    plt.imsave(f'output/label/{code}.png', blank_mask)
    np.save(f'output/label/{code}.npy', blank_mask.astype(int))
    print(f'output/label/{code}.npy are saved')

def add_mask(new, canvas):
    mask_mask = np.invert(np.logical_and(new, canvas))
    canvas = (new >= 1) * new + np.multiply(mask_mask, canvas)
    
    return canvas

def mask_filter(code, elevation, label):
    json_file = f'output/json/{code}_combine.json'
    coco = COCO(json_file)
    img = coco.imgs[1]
    sigma = 1

    blank_mask = np.zeros((img['height'], img['width']))
    for k in list(label_dict.keys()):
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=k, iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        cat_mask = np.zeros((img['height'], img['width']))
        
        # instance wise
        if k in [2, 4, 6, 8]:
            mask = coco.annToMask(anns[0])
            for i, annotation in enumerate(tqdm(anns)):
                mask = coco.annToMask(annotation)
                mask_elevation = elevation * mask * (label == k) 
                mask_elevation = box_filter(mask_elevation, label, sigma, k, 5) * mask
                cat_mask = add_mask(mask_elevation, cat_mask) 

        # if k in [4, 6]:
        #     mask = coco.annToMask(anns[0])
        #     for i, annotation in enumerate(tqdm(anns)):
        #         mask = coco.annToMask(annotation)
        #         mask_elevation = elevation * mask
        #         mask_elevation = box_filter(mask_elevation, label, sigma, k, 5) * mask
        #         cat_mask = add_mask(mask_elevation, cat_mask)        
        #         break        

        # all intances
        if k in [1]:
            cat_mask = elevation * (label == k)
        #     mask_elevation = IQR(mask_elevation.ravel()).reshape(label.shape)
        #     cat_mask = box_filter(mask_elevation, label, sigma, k) * (label == k)
        
        np.save(f'output/mask/{code}_{k}.npy', cat_mask)
        check_mask(cat_mask, f'output/mask/{code}_{k}.png')
        blank_mask = add_mask(cat_mask, blank_mask)    
    
    np.save(f'output/dem/{code}_final.npy', blank_mask)
    return blank_mask

def stick(code_list):
    letters = sorted(set([ c[0] for c in code_list]))
    index = sorted(set([ c[1:] for c in code_list]))
    print(letters)
    print(index)

    json_file = f'output/json/{code_list[0]}_combine.json'
    coco = COCO(json_file)
    img = coco.imgs[1]
    h, w= img['height'], img['width']
    # blank_mask = np.zeros((h * len(index), w * len(letters), 4))
    blank_elevation = np.zeros((h * len(index), w * len(letters)))
    blank_label = np.zeros((h * len(index), w * len(letters)))

    vmin = []
    vmax = []
    
    for i in range(len(code_list)):
        elevation = np.load(f'output/dem/{code_list[i]}.npy')
        vmin.append(np.min(elevation))
        vmax.append(np.max(elevation))
    
    for x, l in enumerate(letters):
        for y, i in enumerate(index):
            code = f'{l}{i}'
            if code in code_list:
                elevation = np.load(f'output/dem/{code}.npy')
                label = np.load(f'output/label/{code}.npy')
                ground = np.load(f'output/mask/{code}_1.npy')

                image = np.asarray(Image.open(f'data/{code}.png')) / 255

                # mask_elevation = mask_filter(code, elevation, label)
                
                # plt.imsave(f'output/dem/{code}_final.png', elevation, vmin=np.min(vmin), vmax=np.max(vmax), cmap='plasma')
                # blank_mask[h*y: h*(y+1), w*x: w*(x+1), :] = image
                blank_elevation[h*y: h*(y+1), w*x: w*(x+1)] = elevation
                blank_label[h*y: h*(y+1), w*x: w*(x+1)] = label

    return blank_elevation, blank_label    

def plot_elevation(code_list):
    vmin = []
    vmax = []

    for i in range(len(code_list)):
        elevation = np.load(f'output/dem/{code_list[i]}_final.npy')
        vmin.append(np.min(elevation))
        vmax.append(np.max(elevation))

    fig, axs = plt.subplots(1, 3, dpi=500, figsize=(10, 4))

    for i in range(len(code_list)):
        ax = axs[i]
        ax.set_title(code_list[i])
        ax.axis('off')
        elevation = np.load(f'output/dem/{code_list[i]}_final.npy')
        im = ax.imshow(elevation, vmin=np.min(vmin), vmax=np.max(vmax))
    
    cbar = fig.colorbar(im, ax=axs, location='bottom')
    cbar.ax.tick_params(labelsize=20)
    plt.savefig('output/test.png')

def plot_label(code_list):
    fig, axs = plt.subplots(1, 3, dpi=500, figsize=(10, 4))

    for i in range(len(code_list)):
        ax = axs[i]
        ax.set_title(code_list[i])
        ax.axis('off')
        elevation = np.load(f'output/label/{code_list[i]}.npy')
        cmap = plt.get_cmap('plasma', 9)
        ax.imshow(elevation, cmap=cmap)

    patch_list = [ patches.Patch(facecolor=c, label=l) for c, l in zip(cmap(np.arange(0, 9, 1)), list(label_dict.values())) ]
    fig.legend(handles=patch_list, loc='lower center', ncol=3)
    plt.savefig('output/test.png')

if __name__ == '__main__':
    code_list = ['A4', 'A5', 'A6', 'B4', 'B5', 'B6', 'C4', 'C5', 'C6', 'C7', 'D4', 'D5', 'D6', 'D7', 'E4', 'E5', 'E6', 'E7']
    code_list = ['B5', 'C5']

    # for code in code_list:
    #     combine_image_id(code)
    #     convert_to_mask(code)
    
    # plot_elevation(code_list)
    # plot_label(code_list)

    elevation, label = stick(code_list)
    plt.imsave(f'output/test.png', elevation)

    check_dem_label(elevation, label.astype(int))