#!/bin/bash
# Converts GT from OID into COCO format


RVC_OBJ_DET_SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
# All data is downloaded to subfolders of RVC_DATA_DIR; if this is not defined: use the root dir + /datasets

RVC_DATA_SRC_DIR=${RVC_DATA_DIR}/


if [ -z "${RVC_JOINED_TRG_DIR}" ]; then
  RVC_DATA_TRG_DIR=${RVC_DATA_SRC_DIR}/
else
  RVC_DATA_TRG_DIR=${RVC_JOINED_TRG_DIR}/
fi

#check if oid has already been converted 
if [ ! -f "$RVC_DATA_SRC_DIR/oid/annotations/openimages_challenge_2019_train_bbox.json" ]; then
  echo "Converting OID to COCO format..."
  # remapping OID format to COCO
  python3 convert_annotations.py --path $RVC_DATA_SRC_DIR/oid/ --version challenge_2019
fi

RVC_DATA_TRG_DIR=
RVC_DATA_SRC_DIR=
RVC_OBJ_DET_SCRIPT_DIR=

echo "Finished remapping."
