# Semantic segmentation reference training scripts

This folder contains reference training scripts for semantic segmentation.
They serve as a log of how to train specific models, as provide baseline
training and evaluation scripts to quickly bootstrap research.

All models have been trained on 8x V100 GPUs.

You must modify the following flags:

`--data-path=/path/to/dataset`

`--nproc_per_node=<number_of_gpus_available>`

## All models
```
2,2,3,3,3:5,5,0,0,5,5,0,0,7,5,7,0,3,5,5,0,7,5,5,0:3,3,0,0,4,3,0,0,4,3,3,0,6,4,4,0,6,6,6,0 pfnag-cpu-30.pt
2,3,3,3,3:5,3,0,0,5,5,7,0,5,7,7,0,5,5,7,0,7,5,7,0:3,4,0,0,4,6,3,0,6,4,6,0,6,6,6,0,6,6,6,0 pfnag-cpu-35.pt
2,3,4,4,4:5,3,0,0,7,5,7,0,5,5,3,3,5,7,7,7,3,3,3,7:4,4,0,0,4,4,4,0,4,6,6,4,6,4,6,6,6,6,6,4 pfnag-cpu-40.pt
2,3,4,4,4:7,5,0,0,7,7,7,0,7,5,3,7,7,7,7,7,7,3,3,7:4,6,0,0,6,6,6,0,6,6,6,6,6,6,6,6,6,6,6,6 pfnag-cpu-45.pt
3,4,4,4,4:5,3,7,0,5,7,7,7,7,7,3,7,7,7,7,7,5,7,3,7:4,4,6,0,6,6,6,6,6,6,4,6,6,6,6,6,6,6,6,6 pfnag-cpu-50.pt

2,2,3,3,4:3,3,0,0,5,3,0,0,5,5,3,0,3,7,3,0,5,3,3,7:3,3,0,0,3,4,0,0,3,3,3,0,4,3,4,0,6,6,6,3 pfnag-gpu-90.pt
2,3,3,4,4:3,3,0,0,5,3,3,0,5,5,3,0,3,5,5,5,7,7,7,5:3,3,0,0,4,4,4,0,4,3,4,0,6,4,6,4,6,6,6,4 pfnag-gpu-115.pt
2,3,4,4,4:3,3,0,0,7,5,5,0,5,7,5,5,3,5,5,5,7,7,5,5:4,3,0,0,6,4,4,0,4,4,6,6,6,6,6,6,6,6,6,4 pfnag-gpu-140.pt
2,4,4,4,4:3,3,0,0,7,5,5,5,7,5,7,5,7,7,7,5,7,7,7,5:3,4,0,0,6,6,6,6,6,6,4,6,6,6,6,6,6,6,6,6 pfnag-gpu-165.pt
3,4,4,4,4:3,3,5,0,7,3,5,7,7,7,5,3,7,5,7,5,7,3,7,3:4,4,6,0,6,4,6,6,6,6,6,6,6,6,6,6,6,6,6,6 pfnag-gpu-190.pt

2,3,3,3,3:3,3,0,0,3,5,3,0,3,3,5,0,3,3,5,0,5,5,7,0:3,3,0,0,4,3,3,0,4,4,4,0,4,6,6,0,6,6,3,0 pfnag-mobile-80.pt
2,3,3,4,4:5,3,0,0,5,3,3,0,3,5,5,0,5,5,5,3,7,5,7,5:3,4,0,0,3,4,4,0,3,3,6,0,4,6,6,6,6,6,6,6 pfnag-mobile-110.pt
2,4,4,4,4:5,3,0,0,5,5,5,3,5,5,5,5,3,5,5,5,7,5,5,5:4,3,0,0,4,4,4,6,6,3,6,3,6,6,4,6,6,6,6,3 pfnag-mobile-140.pt
3,4,4,4,4:5,3,7,0,5,5,5,5,5,5,5,5,3,5,5,3,7,5,5,5:4,4,6,0,4,4,6,6,6,4,6,6,6,6,4,6,6,6,6,6 pfnag-mobile-170.pt
4,4,4,4,4:5,5,5,5,5,7,3,5,7,5,5,3,3,5,5,3,5,3,3,5:6,6,6,6,4,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6 pfnag-mobile-200.pt
```

ours
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --dataset coco -b 4 --aux-loss --wd 0.000001 --arch 3,3,3,4,4:5,5,7,0,5,5,7,0,5,7,5,0,5,5,5,7,5,7,7,7:6,6,3,0,6,4,6,0,6,6,6,0,6,6,6,4,6,6,4,4

LD_LIBRARY_PATH=~/miniconda3/lib python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --aux-loss --arch 4,4,4,4,4:5,5,5,5,5,7,3,5,7,5,5,3,3,5,5,3,5,3,3,5:6,6,6,6,4,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6 --output-dir output | tee output.txt

LD_LIBRARY_PATH=~/miniconda3/lib python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --pretrained --arch 4,4,4,4,4:5,5,5,5,5,7,3,5,7,5,5,3,3,5,5,3,5,3,3,5:6,6,6,6,4,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6 --lr 0.001 --output-dir output | tee output.txt
```