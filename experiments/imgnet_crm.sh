cd ../src/

python3 compute_results.py  \
    --out_folder1 /ssd_scratch/cvit/shyam/crossentropy_tieredimagenet/model_snapshots/checkpoint.epoch0055.pth.tar \
    --out_folder2 /ssd_scratch/cvit/shyam/crossentropy_tieredimagenet/model_snapshots/checkpoint.epoch0060.pth.tar \
    --out_folder3 /ssd_scratch/cvit/shyam/crossentropy_tieredimagenet/model_snapshots/checkpoint.epoch0065.pth.tar \
    --out_folder4 /ssd_scratch/cvit/shyam/crossentropy_tieredimagenet/model_snapshots/checkpoint.epoch0070.pth.tar \
    --out_folder5 /ssd_scratch/cvit/shyam/crossentropy_tieredimagenet/model_snapshots/checkpoint.epoch0075.pth.tar \
    --expname ../logs/imgnet_crm.csv \
    --rerank 1 
