import torch
from kangaroo.kangaroo_model import KangarooModel

print('initializing model...')
model = KangarooModel(
    model_id='lmsys/vicuna-7b-v1.5',
    adapter_mode='none',
    adapter_path=None,
    exit_layer=2,
    dtype='float16'
)
model.attach_drafter_head(r=8, alpha=8.0)
print('Model attached drafter head')
trainable = model.dvi_trainable_params()
print('trainable params length:', len(trainable))
for idx, p in enumerate(trainable):
    print(f'param {idx} shape={tuple(p.shape)} requires_grad={p.requires_grad}')
