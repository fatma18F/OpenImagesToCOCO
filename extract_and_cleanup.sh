#!/bin/bash

RVC_OID_SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
RVC_EXTR_ROOT_DIR=${RVC_DATA_DIR}/


echo "Extracting OID files and removing archive files in the process. This can take some time!"
read -p "This process will remove archives (zip/tar/tar.gz) once they are extracted. Proceed? [y/n] " -n 1 -r RVC_CONFIRM_EXTR
echo    # move to a new line
if [[ ! $RVC_CONFIRM_EXTR =~ ^[Yy]$ ]]; then
  RVC_EXTR_ROOT_DIR=
  RVC_CONFIRM_EXTR=
  exit 1
fi

echo "Extracting OID"
pushd ${RVC_EXTR_ROOT_DIR}/oid
for onetarfile in *.tar.gz
do
  echo "Extracting tar file $onetarfile..."
  tar xf $onetarfile -C ${RVC_EXTR_ROOT_DIR}/oid && rm $onetarfile
done
popd

unzip ${RVC_EXTR_ROOT_DIR}/oid/open-images-object-detection-rvc-2020.zip -d ${RVC_EXTR_ROOT_DIR}/oid
rm ${RVC_EXTR_ROOT_DIR}/oid/open-images-object-detection-rvc-2020.zip
rm ${RVC_EXTR_ROOT_DIR}/oid/sample_submission.csv

RVC_EXTR_ROOT_DIR=
RVC_CONFIRM_EXTR=

echo "Finished extractions."

