#!/bin/bash
for (( i=1; i<=100; i++ )); do
  sbatch inference.sh /home/xi9/code/DATASET/DL3DV-960P-2K-split/folders_part_$i.txt "/home/xi9/code/SVDFor3DGS/model_outputs/DL3DV_fullres_camP-2K-revised-100k-H100x8" "/scratch/xi9/DATASET/DL3DV-960P-2K-SVD/DL3DV_fullres_camP-2K-revised-100k-H100x8"
done