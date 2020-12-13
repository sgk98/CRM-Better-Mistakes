cd ../src/

python3 compute_results.py  \
    --data inaturalist19-224 \
    --batch-size 128 \
    --out_folder1 /ssd_scratch/cvit/shyam/softlabels_inaturalist19_beta20/model_snapshots/checkpoint.epoch0185.pth.tar \
    --out_folder2 /ssd_scratch/cvit/shyam/softlabels_inaturalist19_beta20/model_snapshots/checkpoint.epoch0190.pth.tar \
    --out_folder3 /ssd_scratch/cvit/shyam/softlabels_inaturalist19_beta20/model_snapshots/checkpoint.epoch0195.pth.tar \
    --out_folder4 /ssd_scratch/cvit/shyam/softlabels_inaturalist19_beta20/model_snapshots/checkpoint.epoch0200.pth.tar \
    --out_folder5 /ssd_scratch/cvit/shyam/softlabels_inaturalist19_beta20/model_snapshots/checkpoint.epoch0205.pth.tar \
    --expname ../logs/inat_beta20.csv \
    
