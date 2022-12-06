import os
import csv
import json
import utils
import argparse
import time 
import numpy as np
from dask import dataframe as dd
import tqdm
import json
#import files
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Convert Open Images annotations into MS Coco format')
    parser.add_argument('-p', '--path', dest='path',
                        help='path to openimages data', 
                        type=str)
    parser.add_argument('--version',
                        default='v6',
                        choices=['v4', 'v5', 'v6', 'challenge_2019'],
                        type=str,
                        help='Open Images Version')
    parser.add_argument('--subsets',
                        type=str,
                        nargs='+',
                        default=['val', 'train'],
                        choices=['train', 'val', 'test'],
                        help='subsets to convert')
    
    args = parser.parse_args()
    return args

args = parse_args()
base_dir = args.path
if not isinstance(args.subsets, list):
    args.subsets = [args.subsets]

for subset in args.subsets:
    # Convert annotations
    print('converting {} data'.format(subset))

    # Select correct source files for each subset        
    if subset == 'train' and args.version != 'challenge_2019':
        category_sourcefile = 'class-descriptions-boxable.csv'
        image_sourcefile = 'train-images-boxable-with-rotation.csv'
        if args.version == 'v6':
            annotation_sourcefile = 'oidv6-train-annotations-bbox.csv'
        else:
            annotation_sourcefile = 'train-annotations-bbox.csv'
        image_label_sourcefile = 'train-annotations-human-imagelabels-boxable.csv'

    elif subset == 'val' and args.version != 'challenge_2019':
        category_sourcefile = 'class-descriptions-boxable.csv'
        image_sourcefile = 'validation-images-with-rotation.csv'
        annotation_sourcefile = 'validation-annotations-bbox.csv'
        image_label_sourcefile = 'validation-annotations-human-imagelabels-boxable.csv'
        
    elif subset == 'test' and args.version != 'challenge_2019':
        category_sourcefile = 'class-descriptions-boxable.csv'
        image_sourcefile = 'test-images-with-rotation.csv'
        annotation_sourcefile = 'test-annotations-bbox.csv'
        image_label_sourcefile = 'test-annotations-human-imagelabels-boxable.csv'

    elif subset == 'train' and args.version == 'challenge_2019':
        category_sourcefile = 'challenge-2019-classes-description-500.csv'
        image_sourcefile = 'train-images-boxable-with-rotation.csv'
        annotation_sourcefile = 'challenge-2019-train-detection-bbox.csv'
        image_label_sourcefile = 'challenge-2019-train-detection-human-imagelabels.csv'
        

    elif subset == 'val' and args.version == 'challenge_2019':
        category_sourcefile = 'challenge-2019-classes-description-500.csv'
        image_sourcefile = 'validation-images-with-rotation.csv'
        annotation_sourcefile = 'challenge-2019-validation-detection-bbox.csv'
        image_label_sourcefile = 'challenge-2019-validation-detection-human-imagelabels.csv'
       

    # Load original annotations
    print('loading original annotations ...', end='\r')
    original_category_info = utils.csvread(os.path.join(base_dir, 'annotations', category_sourcefile))
    original_image_metadata = utils.csvread(os.path.join(base_dir, 'annotations', image_sourcefile))
    original_image_annotations = utils.csvread(os.path.join(base_dir, 'annotations', image_label_sourcefile))
    
    #original_annotations = utils.csvread(os.path.join(base_dir, 'annotations', annotation_sourcefile))
    original_annotations_path=f'/mnt/data_4TB/rvc_devkit/datasets/oid/annotations/{annotation_sourcefile}'
    start = time.time()
    
    ddf = dd.read_csv(original_annotations_path,blocksize="25MB")
    #original_annotations=original_annotations.compute().values.tolist()
    cumlens = ddf.map_partitions(len).compute().cumsum()
    original_annotations = [ddf.partitions[0]]
    for npart, partition in enumerate(ddf.partitions[1:].partitions):
         partition.index = partition.index + cumlens[npart]
         original_annotations.append(partition)
   
    end = time.time()
    print("Read csv with dask: ",(end-start),"sec")
    print('loading original annotations ... Done')
    
    tarsier_classes=['Airplane','Bicycle','Bird',  'Boat','Bus','Car','Cat','Cow','Dog','Horse','Motorcycle', 'Person',
     'Traffic light', 'Train', 'Truck', 'Helicopter' ]

    oi = {}

    # Add basic dataset info
    print('adding basic dataset info')
    oi['info'] = {'contributos': 'Vittorio Ferrari, Tom Duerig, Victor Gomes, Ivan Krasin,\
                  David Cai, Neil Alldrin, Ivan Krasinm, Shahab Kamali, Zheyun Feng,\
                  Anurag Batra, Alok Gunjan, Hassan Rom, Alina Kuznetsova, Jasper Uijlings,\
                  Stefan Popov, Matteo Malloci, Sami Abu-El-Haija, Rodrigo Benenson,\
                  Jordi Pont-Tuset, Chen Sun, Kevin Murphy, Jake Walker, Andreas Veit,\
                  Serge Belongie, Abhinav Gupta, Dhyanesh Narayanan, Gal Chechik',
                  'description': 'Open Images Dataset {}'.format(args.version),
                  'url': 'https://storage.googleapis.com/openimages/web/index.html',
                  'version': '{}'.format(args.version),
                  'year': 2020}

    # Add license information
    print('adding basic license info')
    oi['licenses'] = [{'id': 1, 'name': 'Attribution-NonCommercial-ShareAlike License', 'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/'},
                      {'id': 2, 'name': 'Attribution-NonCommercial License', 'url': 'http://creativecommons.org/licenses/by-nc/2.0/'},
                      {'id': 3, 'name': 'Attribution-NonCommercial-NoDerivs License', 'url': 'http://creativecommons.org/licenses/by-nc-nd/2.0/'},
                      {'id': 4, 'name': 'Attribution License', 'url': 'http://creativecommons.org/licenses/by/2.0/'},
                      {'id': 5, 'name': 'Attribution-ShareAlike License', 'url': 'http://creativecommons.org/licenses/by-sa/2.0/'},
                      {'id': 6, 'name': 'Attribution-NoDerivs License', 'url': 'http://creativecommons.org/licenses/by-nd/2.0/'},
                      {'id': 7, 'name': 'No known copyright restrictions', 'url': 'http://flickr.com/commons/usage/'},
                      {'id': 8, 'name': 'United States Government Work', 'url': 'http://www.usa.gov/copyright.shtml'}]



    # Convert category information
    print('converting category info')
    oi['categories'] = utils.convert_category_annotations(original_category_info)
    with open("categories.json", "w") as final:
              json.dump( oi['categories'], final)
    
    tarsier_categories = utils.tarsier_category_annotations(original_category_info,tarsier_classes)
    with open("tarsier_categories.json", "w") as final:
             json.dump( tarsier_categories, final)

    # Convert image mnetadata
    print('converting image info ...')
    image_dir = os.path.join(base_dir, subset)
    oi['images'] =utils.convert_image_annotations(original_annotations, image_dir)
    with open("my_images.json", "w") as final:
              json.dump( oi['images'], final)

    with open("my_images.json", "rt") as final3:
        images = json.load(final3)

   
    print('converting annotations ...')
    # Convert annotations
    utils.convert_Tarsier_instance_annotations(original_annotations,images,oi['categories'],tarsier_classes,tarsier_categories)
    #utils.convert_instance_annotations(original_annotations,images,oi['categories'])

    for i in range(1,9):
       oi =  {}

       with open(f'val_tarsier_classes_annots_{i}.json') as json_file:
          annot = json.load(json_file)
       oi['categories']=tarsier_categories
       oi['images'] =images  
       oi['annotations']=annot
       print(f'writing output to train{i}.json')

       json.dump(oi,  open(f'val_tarsier_train{i}.json', "w")) 
    print('Done')

    
    
    '''  
    # Write annotations into .json file
    filename = os.path.join(base_dir, 'annotations/', 'openimages_{}_{}_{}.json'.format(args.version, subset))
    print('writing output to {}'.format(filename))
    json.dump(oi,  open(filename, "w"))
    print('Done')
    '''