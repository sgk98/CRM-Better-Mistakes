cd ../src/

python3 compute_results.py  \
    --data inaturalist19-224 \
    --batch-size 128 \
    --out_folder1 /ssd_scratch/cvit/shyam/hxe_inaturalist19_alpha0.3/model_snapshots/checkpoint.epoch0225.pth.tar \
    --out_folder2 /ssd_scratch/cvit/shyam/hxe_inaturalist19_alpha0.3/model_snapshots/checkpoint.epoch0230.pth.tar \
    --out_folder3 /ssd_scratch/cvit/shyam/hxe_inaturalist19_alpha0.3/model_snapshots/checkpoint.epoch0235.pth.tar \
    --out_folder4 /ssd_scratch/cvit/shyam/hxe_inaturalist19_alpha0.3/model_snapshots/checkpoint.epoch0240.pth.tar \
    --out_folder5 /ssd_scratch/cvit/shyam/hxe_inaturalist19_alpha0.3/model_snapshots/checkpoint.epoch0245.pth.tar \
    --expname ../logs/inat_alpha03.csv \
    
