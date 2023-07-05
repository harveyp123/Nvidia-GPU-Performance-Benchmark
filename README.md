# Mix Precision Benchmark
We test the performance on FP32, FP16 and TF32 data types

## 1. Environment setup
Create a environment with pytorch in it
```sh
conda create --name torchenv python=3.9
conda activate torchenv
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

## 1. Run code
Run code on the first GPU
```sh
CUDA_VISIBLE_DEVICES=1 python test_mix_precision.py
```

