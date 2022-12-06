#!/bin/sh
# Downloads & extracts OID to RVC_DATA_DIR
# Full dataset sizes: OID: ~527GB

RVC_OBJ_DET_SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
${RVC_OBJ_DET_SCRIPT_DIR}/download_oid_boxable.sh
RVC_OBJ_DET_SCRIPT_DIR =