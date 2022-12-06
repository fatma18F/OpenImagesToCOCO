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
import pandas as pd
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
                        default=['train'],
                        choices=['train', 'val', 'test'],
                        help='subsets to convert')
    parser.add_argument('--myclasses',
                        type=str,
                        nargs='+',
                        default=['COCO'],
                        choices=['COCO', 'Tarsier'],
                        help='choose classes')
    
    args = parser.parse_args()
    return args

args = parse_args()
base_dir = args.path
myclass=args.myclasses[0]


print (f'capturing only instances in {myclass} ')
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
    

    oi = {}
    
    COCO_CLASSES_no = ( "person", "Bicycle", "car", "Motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", 
      "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
      "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
      "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", 
      "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", 
      "vase", "scissors", "teddy bear", "hair drier", "toothbrush", 
    )
    COCO_CLASSES = ["background", "Person", "Bicycle", "Car", "Motorcycle", "Airplane", "Bus", "Train", "Truck", "Boat", "Traffic light", 
      "Fire hydrant", "Stop sign", "!!!Parking meter", "Bench", "Bird", "Cat", "Dog", "!!!Horse", "Sheep", "Cow", "Elephant",
      "Bear", "Zebra", "Giraffe", "Backpack", "Umbrella", "Handbag", "Tie", "Suitcase", "!!Frisbee", "Ski",
      "Snowboard", "Ball", "Kite", "Baseball bat", "Baseball glove", "Skateboard", "Surfboard", "Tennis racket", "Bottle", 
      "wine glass", "Coffee cup", "Fork", "Knife","Spoon", "Bowl", "Banana", "Apple", "Sandwich", "Orange", "Broccoli", "Carrot", 
      "Hot dog", "Pizza", "!!!Donut", "Cake", "Chair", "Couch", "Houseplant", "Bed", "Table", "Toilet", "Television", "Laptop",
       "Mouse", "!!Remote", "Keyboard", "Mobile phone", "Microwave", "Oven", "Toaster", "Sink", "Refrigerator", "Book", "Clock", 
      "Vase", "Scissors", "Teddy bear", "Hair drier", "Toothbrush" 
              ]
    tarsier_classes=['Airplane','Bicycle','Bird', 'Boat','Bus','Car','Cat','Cow','Dog','Horse','Motorcycle', 'Person',
     'Traffic light', 'Train', 'Truck', 'Helicopter' 
    ]
    if myclass=='COCO':
        cls=COCO_CLASSES
    else :
        cls=tarsier_classes
    # Convert category information
    print('converting category info')
    categories_oi = utils.convert_category_annotations(original_category_info)
    oi['categories'] = utils.coco_category_annotations(original_category_info,cls)
    with open("/mnt/data_4TB/rvc_devkit/objdet/openimages2coco/oimages_subset/results/categories_oi_coco.json", "w") as final:
              json.dump( oi['categories'], final)
    
    
    #tarsier_categories = utils.tarsier_category_annotations(original_category_info,tarsier_classes)
    #with open("tarsier_categories.json", "w") as final:
    #         json.dump( tarsier_categories, final)

    # Convert image mnetadata
    print('converting image info ...')
    image_dir = os.path.join(base_dir, subset)
    oi['images'] =utils.convert_image_annotations(original_annotations, image_dir)
    with open("/mnt/data_4TB/rvc_devkit/objdet/openimages2coco/oimages_subset/results/my_images.json", "w") as final:
              json.dump( oi['images'], final)
    
    #with open("my_new_images.json", "rt") as final3:
    #    images = json.load(final3)

   
    print('converting annotations ...')
    # Convert annotations
    oi['annotations']=utils.convert_instance_annotations(original_annotations,oi['images'],categories_oi,oi['categories'],cls)

    '''for i in range(1,9):
       oi =  {}

       with open(f'val_tarsier_classes_annots_{i}.json') as json_file:
          annot = json.load(json_file)
       oi['categories']=tarsier_categories
       oi['images'] =images  
       oi['annotations']=annot
       print(f'writing output to train{i}.json')
    '''

    with open("coco.json", "rt") as final3:
        coco = json.load(final3)
    oi['categories']=coco

    json.dump(oi,  open(f'/mnt/data_4TB/rvc_devkit/objdet/openimages2coco/oimages_subset/results/truth_boxes.json', "w")) 
    print('Done')

    
    
    '''  
    # Write annotations into .json file
    filename = os.path.join(base_dir, 'annotations/', 'openimages_{}_{}_{}.json'.format(args.version, subset))
    print('writing output to {}'.format(filename))
    json.dump(oi,  open(filename, "w"))
    print('Done')
    '''