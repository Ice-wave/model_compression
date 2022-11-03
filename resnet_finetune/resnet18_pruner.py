
import torch
from nni.compression.pytorch.utils import count_flops_params
from torchvision.models import resnet18

from nni.algorithms.compression.v2.pytorch.pruning.basic_pruner import L1NormPruner
from nni.algorithms.compression.v2.pytorch.pruning.basic_pruner import L2NormPruner
from nni.compression.pytorch.speedup import ModelSpeedup


model = resnet18(pretrained=True)


config_list1 = [{
    'sparsity': 0.7,
    'op_types': ['Conv2d']
}]

pruner = L1NormPruner(model, config_list1)
_, masks = pruner.compress()
pruner._unwrap_model()
ModelSpeedup(model, dummy_input=torch.ones((1,3,224,224)), masks_file=masks).speedup_model()



input_size = [1, 3, 224, 224]
device = torch.device('cpu')
dummy_input = torch.randn(input_size).to(device)
flops, params, results = count_flops_params(model, dummy_input)
print(f"Model FLOPs {flops/1e6:.2f}M, Params {params/1e6:.2f}M")

