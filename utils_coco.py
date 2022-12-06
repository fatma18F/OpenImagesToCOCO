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

tarsier_classes=['Airplane','Bicycle','Bird', 'Boat','Bus','Car','Cat','Cow','Dog','Horse','Motorcycle', 'Person',
     'Traffic light', 'Train', 'Truck', 'Helicopter' 
    ]

def coco_category_annotations(orginal_category_info,COCO_CLASSES):
    coco_categories=[]
    
    for i in range(1,len(COCO_CLASSES)):
        cat = {}
        cat['id'] = i 
        cat['name'] = COCO_CLASSES[i]
        '''fid_list= [ row[0] for row in orginal_category_info if row[1]==COCO_CLASSES[i]]
        if not fid_list:
                 fid='no'
                 cat['freebase_id'] = fid
        elif len(fid_list)==1:
             fid=fid_list[0]
             cat['freebase_id'] = fid
        else:
            
            cat['freebase_id'] = fid_list'''


        coco_categories.append(cat)
        
    

    return coco_categories

def convert_image_annotations(new_partitions,
                              image_dir):
    images = []
    added_images=[]
    not_found_images=[]

    print(f'total number of chuncks = {len(new_partitions)}')

    for npartitions in range(0,1):
               #len(new_partitions)):
               print(f'chunk number = {npartitions}')
               for index, ann in tqdm(new_partitions[npartitions].iterrows()):
                   #if index==100:
                   #    return images
                   # Copy information
                   img = {}
                   key=ann['ImageID']
                   if not (key in added_images):
                       filename = os.path.join(image_dir, key + '.jpg')
                       if os.path.isfile(filename):
                             img['width'], img['height'] = imagesize.get(filename)
                       
                             added_images.insert(0,key)
                             img['id'] = key
                             img['file_name'] = key + '.jpg'
                             # Add to list of images
                             images.append(img)
                       
                       else:
                             not_found_images.insert(0,key)
                             #img['width'], img['height'] = (0,0)
                
    print(f'nb of images :{len(images)}')           
    return images


def convert_instance_annotations(new_partitions,images,categories,categories_oi_coco,tarsier_classes):
    
  

    cats_by_freebase_id = {cat['freebase_id']: cat for cat in categories}
    
    imgs = {img['id']: img for img in images}

    annotations = []
    id=0
    for npartitions in range(0,1):
        #len(new_partitions)):
        print( f' {npartitions} from {len(new_partitions)}')
        if npartitions in [5,10,15,20,25,30,35]:
                    print(f' Creating tarsier_classes_annots_{int(npartitions/5)}.json ...' )
                    json.dump(annotations,  open(f'val_tarsier_classes_annots_{int(npartitions/5)}.json', "w"))
                    annotations=[]
        for i, ann in tqdm(new_partitions[npartitions].iterrows()):
       
            # set individual instance id
            # use start_index to separate indices between dataset splits
            annot = {}
            key=ann['ImageID']
            annot['id'] = id
            #annot['image_id'] = key
            #annot['file_name'] = key + '.jpg'
            annot['image_id'] = key
            annot['freebase_id'] = cats_by_freebase_id[ann['LabelName']]['freebase_id']
            annot['category_name'] = cats_by_freebase_id[ann['LabelName']]['name']
            if  annot['category_name'] in tarsier_classes:
                  id+=1
                  coco_id= [ cat['id'] for cat in categories_oi_coco if cat['name']==annot['category_name']][0]
                  annot['category_id']=coco_id
                  #annot['category_id'] = cats_by_freebase_id_coco_categories[ann['LabelName']]['id']
                  if key in imgs:

                       xmin = float(ann['XMin']) * imgs[key]['width']
                       ymin = float(ann['YMin']) * imgs[key]['height']
                       xmax = float(ann['XMax']) * imgs[key]['width']
                       ymax = float(ann['YMax']) * imgs[key]['height']
                       dx = xmax - xmin
                       dy = ymax - ymin
                       annot['bbox'] = [round(a, 2) for a in [xmin , ymin, dx, dy]]
                       annot['iscrowd']=0
                       annot['area'] = round(dx * dy, 2)
                       annotations.append(annot)
    with open("/mnt/data_4TB/rvc_devkit/objdet/openimages2coco/oimages_subset/results/annots_0.json", "w") as final2:
                                  json.dump( annotations, final2)
                   
    return annotations




