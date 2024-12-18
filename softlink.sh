for dir in /scratch/xi9/DATASET/DL3DV-10K-view3/HR/*/; do
    ln -s "$dir" "$(basename "$dir")"
done