import torch
from kangaroo.kangaroo_model import KangarooModel

model = KangarooModel(
    model_id='lmsys/vicuna-7b-v1.5',
    adapter_mode='none',
    adapter_path=None,
    exit_layer=2,
    dtype='float16'
)
model.attach_drafter_head(r=8, alpha=8.0)
trainable = model.dvi_trainable_params()
print('trainable params:', len(trainable))
for idx, p in enumerate(trainable):
    print(idx, p.shape, p.requires_grad)
