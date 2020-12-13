cd ../src/

python3 compute_results.py  \
    --batch-size 128 \
    --out_folder1 /ssd_scratch/cvit/shyam/softlabels_tieredimagenet_beta05/model_snapshots/checkpoint.epoch0095.pth.tar \
    --out_folder2 /ssd_scratch/cvit/shyam/softlabels_tieredimagenet_beta05/model_snapshots/checkpoint.epoch0100.pth.tar \
    --out_folder3 /ssd_scratch/cvit/shyam/softlabels_tieredimagenet_beta05/model_snapshots/checkpoint.epoch0105.pth.tar \
    --out_folder4 /ssd_scratch/cvit/shyam/softlabels_tieredimagenet_beta05/model_snapshots/checkpoint.epoch0110.pth.tar \
    --out_folder5 /ssd_scratch/cvit/shyam/softlabels_tieredimagenet_beta05/model_snapshots/checkpoint.epoch0115.pth.tar \
    --expname ../logs/imgnet_beta04.csv \
    
