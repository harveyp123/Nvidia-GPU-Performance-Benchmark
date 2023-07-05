# Mix Precision Benchmark
We test the performance on FP32, FP16 and TF32 data types

## 1. Environment setup
Create a environment with pytorch in it
```sh
conda create --name torchenv python=3.9
conda activate torchenv
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

## 1. Run code
Create a environment with pytorch in it
```sh
CUDA_VISIBLE_DEVICES=1 python test_mix_precision.py
```

