cd ../src/

python3 compute_results.py  \
    --data inaturalist19-224 \
    --batch-size 128 \
    --out_folder1 /ssd_scratch/cvit/shyam/crossentropy_inaturalist19/model_snapshots/checkpoint.epoch0235.pth.tar \
    --out_folder2 /ssd_scratch/cvit/shyam/crossentropy_inaturalist19/model_snapshots/checkpoint.epoch0240.pth.tar \
    --out_folder3 /ssd_scratch/cvit/shyam/crossentropy_inaturalist19/model_snapshots/checkpoint.epoch0245.pth.tar \
    --out_folder4 /ssd_scratch/cvit/shyam/crossentropy_inaturalist19/model_snapshots/checkpoint.epoch0250.pth.tar \
    --out_folder5 /ssd_scratch/cvit/shyam/crossentropy_inaturalist19/model_snapshots/checkpoint.epoch0255.pth.tar \
    --expname ../logs/inat_crm.csv \
    --rerank 1 
