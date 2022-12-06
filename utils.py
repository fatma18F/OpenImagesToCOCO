import os
import csv
import warnings
import imagesize
import numpy as np
import skimage.io as io
import json 
from tqdm import tqdm
from collections import defaultdict

def csvread(file):
    if file:       
        with open(file, 'r', encoding='utf-8') as f:
            csv_f = csv.reader(f)
            data = []
            for row in csv_f:
                data.append(row)
    else:
        data = None
        
    return data
    

def csvwrite(data, file):
    with open(file, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        for d in data:
            writer.writerow(d)

def _url_to_license(licenses, mode='http'):
    # create dict with license urls as 
    # mode is either http or https
    
    # create dict
    licenses_by_url = {}

    for license in licenses:
        # Get URL
        if mode == 'https':
            url = 'https:' + license['url'][5:]
        else:
            url = license['url']
        # Add to dict
        licenses_by_url[url] = license
        
    return licenses_by_url


def _list_to_dict(list_data):
    
    dict_data = []
    columns = list_data.pop(0)
    for i in range(len(list_data)):
        dict_data.append({columns[j]: list_data[i][j] for j in range(len(columns))})
                         
    return dict_data

def tarsier_category_annotations(orginal_category_info,tarsier_classes):
    
    categories = []
    id=1
    num_categories = len(orginal_category_info)
    for i in range(num_categories):
        cat = {}
        cat['id'] = id
        cat['name'] = orginal_category_info[i][1]
        cat['freebase_id'] = orginal_category_info[i][0]
        if cat['name'] in tarsier_classes:
              categories.append(cat)
              id+=1
    
    return categories


def convert_category_annotations(orginal_category_info):
    
    categories = []
    num_categories = len(orginal_category_info)
    for i in range(num_categories):
        cat = {}
        cat['id'] = i + 1
        cat['name'] = orginal_category_info[i][1]
        cat['freebase_id'] = orginal_category_info[i][0]
        categories.append(cat)
    
    return categories

def convert_image_annotations(new_partitions,
                              image_dir):
    images = []
    added_images=[]
    print(f'total number of chuncks = {len(new_partitions)}')

    for npartitions in range(0,len(new_partitions)):
               print(f'chunk number = {npartitions}')
               for index, ann in tqdm(new_partitions[npartitions].iterrows()):
                   # Copy information
                   img = {}
                   key=ann['ImageID']
                   if not (key in added_images):
                       
                       added_images.insert(0,key)
                       img['id'] = key
                       img['file_name'] = key + '.jpg'
       
                       image_size_dict = {}
                       filename = os.path.join(image_dir, key + '.jpg')
                       if os.path.isfile(filename):
                             img['width'], img['height'] = imagesize.get(filename)
                       else:
                             img['width'], img['height'] = (0,0)

                       # Add to list of images
                       images.append(img)
                
    return images


def convert_Tarsier_instance_annotations(new_partitions,images,categories,tarsier_classes,tarsier_categories):
    
    cats_by_freebase_id = {cat['freebase_id']: cat for cat in categories}
    cats_by_freebase_id_tarsier_categories= {cat['freebase_id']: cat for cat in tarsier_categories}
    
    imgs = {img['id']: img for img in images}

    annotations = []
    id=0
    for npartitions in range(0,len(new_partitions)):
        print( f' {npartitions} from {len(new_partitions)}')
        if npartitions in [5,10,15,20,25,30,35]:
                    print(f' Creating tarsier_classes_annots_{int(npartitions/5)}.json ...' )
                    json.dump(annotations,  open(f'tarsier_classes_annots_{int(npartitions/5)}.json', "w"))
                    annotations=[]
        for i, ann in tqdm(new_partitions[npartitions].iterrows()):
       
            # set individual instance id
            # use start_index to separate indices between dataset splits
            annot = {}
            key=ann['ImageID']
            #annot['image_id'] = key
            #annot['file_name'] = key + '.jpg'
            annot['image_id'] = key
            annot['freebase_id'] = cats_by_freebase_id[ann['LabelName']]['freebase_id']
            annot['category_name'] = cats_by_freebase_id[ann['LabelName']]['name']
            if  annot['category_name'] in tarsier_classes:
                  annot['id'] = id
                  id+=1
                  annot['category_id'] = cats_by_freebase_id_tarsier_categories[ann['LabelName']]['id']

                  xmin = float(ann['XMin']) * imgs[key]['width']
                  ymin = float(ann['YMin']) * imgs[key]['height']
                  xmax = float(ann['XMax']) * imgs[key]['width']
                  ymax = float(ann['YMax']) * imgs[key]['height']
                  dx = xmax - xmin
                  dy = ymax - ymin
                  annot['bbox'] = [round(a, 2) for a in [xmin , ymin, dx, dy]]
                  annot['area'] = round(dx * dy, 2)
                  annotations.append(annot)
    with open("tarsier_classes_annots_8.json", "w") as final2:
                                  json.dump( annotations, final2)
                   
    #return annotations
def convert_instance_annotations(new_partitions,images,categories):
    
    cats_by_freebase_id = {cat['freebase_id']: cat for cat in categories}    
    imgs = {img['id']: img for img in images}

    annotations = []
    id=0
    for npartitions in range(0,len(new_partitions)):
        print( f' {npartitions} from {len(new_partitions)}')
        if npartitions in [5,10,15,20,25,30,35]:
                    print(f' Creating classes_annots_{int(npartitions/5)}.json ...' )
                    json.dump(annotations,  open(f'classes_annots_{int(npartitions/5)}.json', "w"))
                    annotations=[]
        for i, ann in tqdm(new_partitions[npartitions].iterrows()):
       
            # set individual instance id
            # use start_index to separate indices between dataset splits
            annot = {}
            key=ann['ImageID']
            #annot['image_id'] = key
            #annot['file_name'] = key + '.jpg'
            annot['image_id'] = key
            annot['freebase_id'] = cats_by_freebase_id[ann['LabelName']]['freebase_id']
            annot['category_name'] = cats_by_freebase_id[ann['LabelName']]['name']
            annot['id'] = i
            annot['category_id'] = cats_by_freebase_id[ann['LabelName']]['id']

            xmin = float(ann['XMin']) * imgs[key]['width']
            ymin = float(ann['YMin']) * imgs[key]['height']
            xmax = float(ann['XMax']) * imgs[key]['width']
            ymax = float(ann['YMax']) * imgs[key]['height']
            dx = xmax - xmin
            dy = ymax - ymin
            annot['bbox'] = [round(a, 2) for a in [xmin , ymin, dx, dy]]
            annot['area'] = round(dx * dy, 2)
            annotations.append(annot)
    with open("classes_annots_8.json", "w") as final2:
                                  json.dump( annotations, final2)


''' 
def filter_images(images, annotations):
    image_ids = list(np.unique([ann['image_id'] for ann in annotations]))
    filtered_images = [img for img in images if img['id'] in image_ids]
    return filtered_images
    
'''


