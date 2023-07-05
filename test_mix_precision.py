import torch
from timm.models.vision_transformer import VisionTransformer
from timm.models import create_model
import time
from torch.cuda.amp import GradScaler, autocast

def train(model, input, amp_enable=False):
    torch.cuda.synchronize()
    time_start=time.time()
    dtype = torch.float16
    # dtype = torch.bfloat16
    for i in range(0, 5):
        with autocast(enabled=amp_enable, dtype=dtype):
            out=model(input)
        out.sum().backward()

    torch.cuda.synchronize()
    print(f'time used: {time.time()-time_start}')

if __name__=='__main__':
    model_name = ['resnet50', 'vit_small_patch16_384',  'vit_large_patch16_384', 'resnet34']
    model=create_model(model_name[2],
                       pretrained=False,
                       num_classes=None,
                       drop_rate=0,
                       drop_path_rate=0.3)
    model.cuda().train()
    batch_size = 64 #256
    input=torch.rand(batch_size,3,384,384).cuda()

    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False

    # warmup, ignore
    print('----Warmup----')
    train(model, input)




    print('----train with fp32----')
    train(model, input)
    print('----train with fp16 autocast----')
    train(model, input, amp_enable=True)
    print('----train with tf32----')
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    train(model, input)
