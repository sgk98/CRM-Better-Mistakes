cd ../src/

python3 compute_results.py  \
    --batch-size 128 \
    --out_folder1 /ssd_scratch/cvit/shyam/hxe_tieredimagenet_alpha0.6/model_snapshots/checkpoint.epoch0040.pth.tar \
    --out_folder2 /ssd_scratch/cvit/shyam/hxe_tieredimagenet_alpha0.6/model_snapshots/checkpoint.epoch0045.pth.tar \
    --out_folder3 /ssd_scratch/cvit/shyam/hxe_tieredimagenet_alpha0.6/model_snapshots/checkpoint.epoch0050.pth.tar \
    --out_folder4 /ssd_scratch/cvit/shyam/hxe_tieredimagenet_alpha0.6/model_snapshots/checkpoint.epoch0055.pth.tar \
    --out_folder5 /ssd_scratch/cvit/shyam/hxe_tieredimagenet_alpha0.6/model_snapshots/checkpoint.epoch0060.pth.tar \
    --expname ../logs/imgnet_alpha06.csv \
    
