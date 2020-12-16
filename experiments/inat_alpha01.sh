cd ../src/

python3 compute_results.py  \
    --data inaturalist19-224 \
    --batch-size 128 \
    --out_folder1 /ssd_scratch/cvit/shyam/hxe_inaturalist19_alpha0.1/model_snapshots/checkpoint.epoch0250.pth.tar \
    --out_folder2 /ssd_scratch/cvit/shyam/hxe_inaturalist19_alpha0.1/model_snapshots/checkpoint.epoch0255.pth.tar \
    --out_folder3 /ssd_scratch/cvit/shyam/hxe_inaturalist19_alpha0.1/model_snapshots/checkpoint.epoch0260.pth.tar \
    --out_folder4 /ssd_scratch/cvit/shyam/hxe_inaturalist19_alpha0.1/model_snapshots/checkpoint.epoch0265.pth.tar \
    --out_folder5 /ssd_scratch/cvit/shyam/hxe_inaturalist19_alpha0.1/model_snapshots/checkpoint.epoch0270.pth.tar \
    --expname ../logs/inat_alpha01.csv \
    
