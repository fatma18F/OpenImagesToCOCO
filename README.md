# Open Image - Object Detection dataset #

our dataset:[OID](https://storage.googleapis.com/openimages/web/index.html)



## Requirements ##
Install additional requirements with:
   - ``` pip install -r requirements.txt ```
   -  ``` pip install "dask[dataframe]" ```


# Functionality

## Dataset Download ##

1. Specify the target directory where you want to download OID dataset: 

``` export RVC_DATA_DIR=/path/to/rvc_dataroot  ```

2. Execute the download script ``` bash download_obj_det.sh ``` which will download OID dataset. The extracted dataset needs ca. 530GB of disk space  

3. download the OpenImages test data from kaggle. [Kaggle test Open Image](https://www.kaggle.com/competitions/google-ai-open-images-object-detection-track/data)

4. After successfully downloading all datasets, execute ``` bash extract_and_cleanup.sh ``` to extract and delete zip files.
 Note, that the script will create and move the different annotation files to `annotations` folder.


## Open image annotations to COCO format convertor ##
- `convert_annotations.py` will load the original .csv annotation files from Open Images, convert the annotations into the list/dict based format of [MS Coco annotations](http://cocodataset.org/#format-data) and store them as a .json files in the same folder.

you can find OID images in `/mnt/data_4TB/rvc_devkit/dataset`

-> Run conversion of bounding box annotations:
```
bash convert_oid_coco.sh 
```
The toolkit supports multiple versions of the dataset including `v4`, `v5`, `v6` and `challenge_2019`.
For example the `bbox` annotations of `challenge_2019` can be converted like:
```
python3 convert_annotations.py --path $RVC_DATA_SRC_DIR/oid/ --version challenge_2019 
```
or
```
python3 convert_annotations.py --path /mnt/data_4TB/rvc_devkit/dataset/oid/ --version challenge_2019 
```
-> The above step creates 8 training and a separate joint validation json files in COCO Object Detection format (only bbox entries, without "segmentation" entries)
Specify the subset you want to convert with the flag  --subset. 
For example the train `bbox` annotations of `challenge_2019` can be converted like:
```
python3 convert_annotations.py --path $RVC_DATA_SRC_DIR/oid/ --version challenge_2019 --subset train
```

## Dataset filtering ##

Filter classes

``` 
Tarsier_classes=['Airplane','Bicycle','Bird',  'Boat','Bus','Car','Cat','Cow','Dog','Horse','Motorcycle', 'Person','Traffic light', 'Train', 'Truck', 'Helicopter' ] 
``` 

```
COCO_CLASSES = ["background", "Person", "Bicycle", "Car", "Motorcycle", "Airplane", "Bus", "Train", "Truck", "Boat", "Traffic light", "Fire hydrant", "Stop sign", "!!!Parking meter", "Bench", "Bird", "Cat", "Dog", "!!!Horse", "Sheep", "Cow", "Elephant","Bear", "Zebra", "Giraffe", "Backpack", "Umbrella", "Handbag", "Tie", "Suitcase", "!!Frisbee", "Ski","Snowboard", "Ball", "Kite", "Baseball bat", "Baseball glove", "Skateboard", "Surfboard", "Tennis racket", "Bottle","wine glass", "Coffee cup", "Fork", "Knife","Spoon", "Bowl", "Banana", "Apple", "Sandwich", "Orange", "Broccoli", "Carrot", "Hot dog", "Pizza", "!!!Donut", "Cake", "Chair", "Couch", "Houseplant", "Bed", "Table", "Toilet", "Television", "Laptop","Mouse", "!!Remote", "Keyboard", "Mobile phone", "Microwave", "Oven", "Toaster", "Sink", "Refrigerator", "Book", "Clock", "Vase", "Scissors", "Teddy bear", "Hair drier", "Toothbrush"  ]
```

Open image dataset has 500 classes.

if you are only interested in one of the above Tarsier or COCO objects select the class `--myclasses COCO` or `--myclasses Tarsier` 

For example:
 ```
python3  filter_OID/convert_annotations_coco.py --path /mnt/data_4TB/rvc_devkit/datasets/oid/ --version challenge_2019 --subsets train --myclasses COCO
```
