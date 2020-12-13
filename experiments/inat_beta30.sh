cd ../src/

python3 compute_results.py  \
    --data inaturalist19-224 \
    --batch-size 128 \
    --out_folder1 /ssd_scratch/cvit/shyam/softlabels_inaturalist19_beta30/model_snapshots/checkpoint.epoch0160.pth.tar \
    --out_folder2 /ssd_scratch/cvit/shyam/softlabels_inaturalist19_beta30/model_snapshots/checkpoint.epoch0165.pth.tar \
    --out_folder3 /ssd_scratch/cvit/shyam/softlabels_inaturalist19_beta30/model_snapshots/checkpoint.epoch0170.pth.tar \
    --out_folder4 /ssd_scratch/cvit/shyam/softlabels_inaturalist19_beta30/model_snapshots/checkpoint.epoch0185.pth.tar \
    --out_folder5 /ssd_scratch/cvit/shyam/softlabels_inaturalist19_beta30/model_snapshots/checkpoint.epoch0180.pth.tar \
    --expname ../logs/inat_beta30.csv \
    
