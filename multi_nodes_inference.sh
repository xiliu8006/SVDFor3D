#!/bin/bash
for (( i=1; i<=100; i++ )); do
  sbatch inference.sh /home/xi9/code/DATASET/DL3DV-960P-2K-split/folders_part_$i.txt "/home/xi9/code/SVDFor3D/model_outputs/DL3DV_fullres_camP-2K-revised-200k-H100x4/checkpoint-210000" "/scratch/xi9/DATASET/DL3DV-960P-2K-SVD/DL3DV_fullres_camP-2K-revised-200k-H100x4"
done