#!/usr/bin/env bash
set -euo pipefail

CHUNK_SIZE=${CHUNK_SIZE:-5000}   
NUM_WORKERS=${NUM_WORKERS:-16}     
DATASET_NAME="egoexo4d_dataset"   

BUILDER_DIR="/path/to/builder" # path to builder.py
TRAIN_DIR="/path/to/train"                     # path to data/train/*.npy
VAL_DIR="/path/to/val"
OUT_ROOT="/path/to/output"                                 # path to rlds dataset
OUT_DATA_DIR="${OUT_ROOT}/${DATASET_NAME}"

cd "${BUILDER_DIR}"

mapfile -t FILES < <(find "${TRAIN_DIR}" -maxdepth 1 -type f -name "*.npy" -printf "%f\n" | sort)
TOTAL=${#FILES[@]}
if (( TOTAL == 0 )); then
  echo "No .npy files under ${TRAIN_DIR}"; exit 1
fi

NUM_CHUNKS=$(( (TOTAL + CHUNK_SIZE - 1) / CHUNK_SIZE ))
echo "Total files: ${TOTAL} | Chunks: ${NUM_CHUNKS} (size=${CHUNK_SIZE})"

for ((i=0; i<NUM_CHUNKS; i++)); do
  start=$(( i * CHUNK_SIZE ))
  end=$(( start + CHUNK_SIZE )); (( end > TOTAL )) && end=$TOTAL

  echo "==> Chunk $((i+1)) / ${NUM_CHUNKS}  [${start} .. $((end-1))]"
  rm -rf "${VAL_DIR}"
  mkdir -p "${VAL_DIR}"

  tmp_list="$(mktemp)"
  for ((j=start; j<end; j++)); do
    echo "${FILES[j]}" >> "${tmp_list}"
  done

  rsync -a --files-from="${tmp_list}" "${TRAIN_DIR}/" "${VAL_DIR}/"
  rm -f "${tmp_list}"

  TFDS_DATA_DIR="${OUT_ROOT}" tfds build --overwrite \
    --beam_pipeline_options="direct_running_mode=multi_processing,direct_num_workers=${NUM_WORKERS}"

  SPLIT_DIR="${OUT_DATA_DIR}/egoexo4d_split_$((i+1))"
  mkdir -p "${SPLIT_DIR}"
  mv "${OUT_DATA_DIR}/1.0.0" "${SPLIT_DIR}/1.0.0"

  rm -rf "${VAL_DIR}"
done

echo "Done. See: ${OUT_DATA_DIR}/egoexo4d_split_*/1.0.0"
