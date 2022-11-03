# CUDA version >= 11.0
# TensorRT version >= 7.2

from torch.optim import SGD
from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer
import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from nni.compression.pytorch.utils import count_flops_params
from torchvision.models import resnet18
from nni.algorithms.compression.v2.pytorch.pruning.basic_pruner import L1NormPruner
from nni.compression.pytorch.speedup import ModelSpeedup
from fer import FER2013

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

resnet_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ]),
}

continue_train = False
epoches = 2
batch_size = 20

torch.backends.cudnn.benchmark = True

trainset = FER2013(split='Training', transform=resnet_data_transforms['train'])
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
PublicTestset = FER2013(
    split='PublicTest', transform=resnet_data_transforms['val'])
PublicTestloader = torch.utils.data.DataLoader(
    PublicTestset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)


train_dataset_size = len(trainset)
val_dataset_size = len(PublicTestset)

print('train_dataset_size', train_dataset_size)
print('val_dataset_size', val_dataset_size)



model_ft = resnet18(pretrained=True)
config_list = [{
    'quant_types': ['input', 'weight'],
    'quant_bits': {'input': 8, 'weight': 8},
    'op_types': ['Conv2d']
}, {
    'quant_types': ['output'],
    'quant_bits': {'output': 8},
    'op_types': ['ReLU']
}]


device = torch.device('cpu')
optimizer = SGD(model_ft.parameters(), 1e-2)

dummy_input = torch.rand(1, 3, 224, 224).to(device)
quantizer = QAT_Quantizer(model_ft, config_list, optimizer, dummy_input)
quantizer.compress()
print(model_ft)



num_fc_ftr = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_fc_ftr, 7)
model_ft = model_ft.to(DEVICE)


criterion = nn.CrossEntropyLoss()
lr = 0.001 / 10
fc_params = list(map(id, model_ft.fc.parameters()))
base_params = filter(lambda p: id(p) not in fc_params, model_ft.parameters())
optimizer = torch.optim.Adam([{'params': base_params}, {
                             'params': model_ft.fc.parameters(), 'lr': lr * 10}], lr=lr, betas=(0.9, 0.999))
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

device = DEVICE
train_acc = []
val_acc = []
best_model = model_ft.state_dict()
best_acc = 0


for epoch in range(epoches):
    model_ft.train()
    iteration = 0
    train_correct = 0
    for batch_idx, data in enumerate(trainloader):
        x, y = data
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        y_hat = model_ft(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        iteration += 1
        # get the index of the max log-probability
        pred = y_hat.max(1, keepdim=True)[1]
        train_correct += pred.eq(y.view_as(pred)).sum().item()
    print('epoch', epoch, 'train accuracy', train_correct/train_dataset_size)
    train_acc.append(train_correct/train_dataset_size)

    model_ft.eval()
    test_loss = 0
    test_correct = 0
    with torch.no_grad():
        for i, data in enumerate(PublicTestloader):
            x, y = data
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            y_hat = model_ft(x)
            test_loss += criterion(y_hat, y).item()  # sum up batch loss
            # get the index of the max log-probability
            pred = y_hat.max(1, keepdim=True)[1]
            test_correct += pred.eq(y.view_as(pred)).sum().item()

    test_loss /= len(PublicTestloader.dataset)
    acc = test_correct / val_dataset_size
    if acc > best_acc:
        best_acc = acc
        best_model = model_ft.state_dict()
    val_acc.append(acc)

    scheduler.step()
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, test_correct, len(PublicTestset), 100. * acc))

# best_model = model_ft.state_dict()
# torch.save(model_ft.state_dict(), os.path.join('./output', "test1.pkl"))



os.makedirs('log', exist_ok=True)
model_path = "./log/fer2013_model1.pth"
calibration_path = "./log/fer2013_calibration1.pth"
calibration_config = quantizer.export_model(model_path, calibration_path)
print("calibration_config: ", calibration_config)

from nni.compression.pytorch.quantization_speedup import ModelSpeedupTensorRT
input_shape = (1, 3, 224, 224)
engine = ModelSpeedupTensorRT(model_ft, input_shape, config=calibration_config, batchsize=20)
engine.compress()


input_size = [1, 3, 224, 224]
device = torch.device('cpu')
dummy_input = torch.randn(input_size).to(device)
flops, params, results = count_flops_params(model_ft, dummy_input)
print(f"Model FLOPs {flops/1e6:.2f}M, Params {params/1e6:.2f}M")
