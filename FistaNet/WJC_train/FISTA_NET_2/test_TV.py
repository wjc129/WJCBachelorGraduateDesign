import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split , SubsetRandomSampler, Subset
from tqdm import tqdm
import torch.nn.functional as F
import CCTProj
from torch.nn import init
import os
import matplotlib.pyplot as plt
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
# initialize the cuda device index
device_ids = [0]
# initialize the num of layers
blocknums = 1
# initialize the size of projection data
Angle_Nums,Proj_Height,Proj_Width = [72,160,800]
# initialize the size of volume data
VolumeHeight, VolumeDepth,VolumeWidth = [40,256,256]
# initialize a tensor : volumeSize
volumeSize = torch.tensor([VolumeHeight, VolumeDepth,VolumeWidth], dtype=torch.int).to(device_ids[0])
# initialize a tensor : projectSize
projectSize = torch.tensor([Proj_Height,Proj_Width], dtype=torch.int).to(device_ids[0])
# initialize the directory save the result after model
result_vol_dir =r"E:\WJC_train\After_train"
#result_vol_dir = "/root/autodl-tmp/WJC_train/After_train"
# initialize a geomeotry tensor:projVector
projectorPath = r'D:\Documents\Downloads\geometry_sparse_72.raw'
#projectorPath = "/root/autodl-tmp/WJC_train/geometry_sparse_72.raw"
geometry = np.fromfile(projectorPath, dtype=np.float32)[0:Angle_Nums*12]
projVector = torch.from_numpy(geometry.reshape(Angle_Nums, 12)).to(device_ids[0])
# initialize the location of the volume
#z_bias,dsample,dslice
volinfo = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float).to(device_ids[0])
# initialize volume
volume = torch.zeros(1,VolumeHeight, VolumeDepth,VolumeWidth).to(device_ids[0])
# initialize volume_p
volume_P = torch.zeros(VolumeHeight, VolumeDepth,VolumeWidth).to(device_ids[0])
# load sparse CT projection file path and save in an array
#CT_Proj_dir = "/root/autodl-tmp/WJC_train/train40proj72"
CT_Proj_dir = r"E:\train40proj72"
CT_Proj_Paths = []
Proj_Files = [f for f in os.listdir(CT_Proj_dir) if os.path.isfile(os.path.join(CT_Proj_dir, f))]
for num in range(len(Proj_Files)):
    CT_Proj_Paths.append(os.path.join(CT_Proj_dir,Proj_Files[num]))

# load label 360 degree projection result
#Label_dir = "/root/autodl-tmp/WJC_train/train40label"
Label_dir = r"E:\train40label"

volume_init_dir = r"E:\train40init"
#volume_init_dir = "/root/autodl-tmp/WJC_train/train40init"
Label_Files = [f for f in os.listdir(Label_dir) if os.path.isfile(os.path.join(Label_dir, f))]
Label_Paths = []
volume_init_Files = [f for f in os.listdir(volume_init_dir) if os.path.isfile(os.path.join(volume_init_dir, f))]
volume_init_paths = []


for num in range(len(Label_Files)):
    Label_Paths.append(os.path.join(Label_dir,Label_Files[num]))

for num in range(len(volume_init_Files)):
    volume_init_paths.append(os.path.join(volume_init_dir,volume_init_Files[num]))


class SumLossFunc(nn.Module):
    def __init__(self, weight: float = 0.01):
        super(SumLossFunc, self).__init__()
        self.loss = nn.L1Loss()
        self.weight = weight

    def forward(self, output,target):
        layernum = target.size(1)
        loss1 = self.loss(output[:, 3:layernum-4 :, :],target[:, 3:layernum-4, :,:])
        _, c, h, w = output.size()
        tv_h = torch.abs(output[:, 3:layernum-4, 1:, :] - output[:, 3:layernum-4, :-1, :]).sum()
        tv_w = torch.abs(output[:, 3:layernum-4, :, 1:] - output[:, 3:layernum-4, :, :-1]).sum()
        loss2 = self.weight * (tv_h + tv_w) / (c * h * w)
        loss = loss1 + 0.0001*loss2

        return loss

class CTDataProjectionDataset(Dataset):
    def __init__(self, projection_paths, label_paths,volume_init_paths):
        self.projection_paths = projection_paths
        self.label_paths = label_paths
        self.volume_init_paths = volume_init_paths

    def __len__(self):
        return len(self.projection_paths)

    def __getitem__(self, idx):
        projection_path = self.projection_paths[idx]
        label_path = self.label_paths[idx]
        volume_init_path = self.volume_init_paths[idx]
        projection_data  = torch.from_numpy(np.fromfile(projection_path, dtype=np.float32)).reshape(Angle_Nums,Proj_Height,Proj_Width).to(device_ids[0])
        label = torch.from_numpy(np.fromfile(label_path, dtype=np.float32)).reshape(VolumeHeight, VolumeDepth,VolumeWidth).to(device_ids[0])
        volume_init = torch.from_numpy(np.fromfile(volume_init_path, dtype=np.float32)).reshape(VolumeHeight, VolumeDepth,VolumeWidth).to(device_ids[0])
        return projection_data, label,volume_init




sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(device_ids[0])
sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).to(device_ids[0])
#loss_function = gradientloss()
loss_function = SumLossFunc()
# create the CTDataProjectionDataset instance
dataset = CTDataProjectionDataset(CT_Proj_Paths, Label_Paths,volume_init_paths)

train_size = int(0.8*len(dataset))
val_size = int(0.1* len(dataset))
test_size = int(0.1*len(dataset))
indices = list(range(len(dataset)))

train_indices = indices[:train_size]
val_indices = indices[train_size:(train_size + val_size)]
test_indices = indices[(train_size + val_size):(train_size + val_size)+test_size]
'''
train_indices = indices[:train_size]
val_indices = indices[:train_size]
test_indices = indices[:train_size]
'''

# create dataluoader
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(dataset, batch_size=1, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=1, sampler=val_sampler)
test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler)


iterationtime = 4

# define the train model
class FistaNet_Block(nn.Module):
    def __init__(self,features,max_value,min_value):
        super(FistaNet_Block, self).__init__()
        self.CTForward = CCTProj.FProj()
        self.CTBackward = CCTProj.BProj()
        self.theta = nn.Parameter(torch.Tensor([0.0]).to(device_ids[0]))
        self.miu = nn.Parameter(torch.Tensor([1.0]).to(device_ids[0]))
        self.rou = nn.Parameter(torch.Tensor([-1.0]).to(device_ids[0]))
        self.max_value = max_value
        self.min_value = min_value
        self.Sp = nn.Softplus()

        self.conv_D = nn.Conv3d(1, features, (1, 1, 1), stride=1, padding=0)
        self.conv1_forward = nn.Conv3d(features, features, (3, 3, 3), stride=1, padding=1)
        self.conv2_forward = nn.Conv3d(features, features, (3, 3, 3), stride=1, padding=1)
        self.conv3_forward = nn.Conv3d(features, features, (3, 3, 3), stride=1, padding=1)
        self.conv4_forward = nn.Conv3d(features, features, (3, 3, 3), stride=1, padding=1)

        self.conv1_backward = nn.Conv3d(features, features, (3, 3, 3), stride=1, padding=1)
        self.conv2_backward = nn.Conv3d(features * 2, features, (3, 3, 3), stride=1, padding=1)
        self.conv3_backward = nn.ConvTranspose3d(features * 2, features, (3, 3, 3), stride=1, padding=1)
        self.conv4_backward = nn.ConvTranspose3d(features * 2, features, (3, 3, 3), stride=1, padding=1)
        self.conv_G = nn.Conv3d(features * 2, 1, (1, 1, 1), stride=1, padding=0)

        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
    def forward(self,volume,origin,sinogram_real, projVector, volumeSize, projSize, volinfo, volume_P):
        volume = volume.squeeze(0)
        sinogram_real = sinogram_real.squeeze(0)
        sinogram = self.CTForward(input=volume, sinogram_real=sinogram_real, projVector=projVector,
                                  volumeSize=volumeSize, projSize=projSize, volinfo=volinfo)
        residul = sinogram - sinogram_real
        volume_back = self.CTBackward(residual=residul, projVector=projVector, volumeSize=volumeSize, projSize=projSize,
                                      volinfo=volinfo, volume_P=volume_P)
        volume_back_d = volume - self.miu * volume_back
        volume = volume_back_d.unsqueeze(0)
        volume = volume.unsqueeze(1)
        volume_to1 = (volume - torch.min(volume)) / (torch.max(volume)-torch.min(volume))
        x_input = volume_to1  # 1 1.0
        x_D = self.conv_D(x_input)  # 32 1.0
        l1 = self.conv1_forward(x_D)  # 32 0.2
        l1 = self.relu(l1)
        l2 = self.conv2_forward(l1)  # 32 0.5
        l2 = self.relu(l2)
        l3 = self.conv3_forward(l2)  # 32 0.25
        l3 = self.relu(l3)
        x_forward = self.conv4_forward(l3)  # 32 0.25
        x_forward_thrld = torch.sign(x_forward) * torch.clamp(torch.abs(x_forward) - self.theta, min=0.0)
        l1b = self.conv1_backward(x_forward_thrld)  # 32 0.25
        l1b = self.relu(l1b)

        l1b = torch.cat([l3, l1b], 1)  # 64 0.25
        l2b = self.conv2_backward(l1b)  # 32 0.25
        l2b = self.relu(l2b)
        l2b = torch.cat([l2, l2b], 1)  # 64 0.5
        l3b = self.conv3_backward(l2b)  # 32 0.5
        l3b = self.relu(l3b)

        l3b = torch.cat([l1, l3b], 1)  # 64 0.5
        x_backward = self.conv4_backward(l3b)  # 32 0.5
        x_backward = self.relu(x_backward)

        x_backward = torch.cat([x_D, x_backward], 1)  # 64 1.0
        x_G = self.conv_G(x_backward)  # 1 1.0

        # prediction output (skip connection);
        x_pred = self.relu(x_input + x_G)
        volume_back_to_ori = (x_pred) * (self.max_value - self.min_value) + self.min_value
        x_pred = volume_back_to_ori.squeeze(0)
        y_pred = x_pred+ self.rou*(x_pred-origin)
        #x_pred.detach().cpu().numpy().astype(np.float32).tofile("x_pred.raw")
        return y_pred,x_pred
num_epochs = 50
# the dataset of the pic -- size-range
max_value = -10000000000
min_value = 10000000000
lable_Files = [f for f in os.listdir(Label_dir) if os.path.isfile(os.path.join(Label_dir, f))]
for num in range(len(lable_Files)):
    lable_path = os.path.join(Label_dir,lable_Files[num])
    raw_file = np.fromfile(lable_path, dtype=np.float32)
    if np.max(raw_file)>=max_value:
        max_value = np.max(raw_file)
    if np.min(raw_file)<=min_value:
        min_value = np.min(raw_file)
max_value = 1800
min_value = 0
lossfile_lists = []
features = 16
model = FistaNet_Block(features,max_value,min_value).to(device_ids[0])
model.load_state_dict(torch.load(r'E:\WJC_train\FISTA_NET_2\训练结果1.0\4轮学习率1e-4\FistaNet_best.pth'))
#model.load_state_dict(torch.load('E:\WJC_train\FISTA_NET_2\学习率1e-4\FistaNet_best.pth'))
#optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#cosine_annealing_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.00001)
# initialize a loss function list
#loss_function = nn.L1Loss()
x = np.arange(1,num_epochs+1)
train_loss_list = np.zeros(num_epochs)
val_loss_list = np.zeros(num_epochs)
val_loss_min = 1000000000


model.eval()
num = 0
# 然后，对数据集中的元素进行推理
test_loss_sum = 0.0
test_loss =0.0
index = 0
with torch.no_grad():

    for sinogram_real, labels ,volume_init in tqdm(test_loader, desc="Testing"):
        index+=1
        volume = volume_init
        origin = volume_init
        losslist = []
        loss0 = loss_function( (volume_init - min_value) / (max_value - min_value), (labels - min_value) / (max_value - min_value))
        losslist.append(loss0.item())
        begintime= time.time()
        for num in range(iterationtime):
            outputs,x_pred= model(volume,origin,sinogram_real, projVector, volumeSize, projectSize, volinfo, volume_P)
            outputs_to1 = (outputs - min_value) / (max_value - min_value)
            labels_to1 = (labels - min_value) / (max_value - min_value)
            loss = loss_function(outputs_to1,labels_to1)
            losslist.append(loss.item())
            #volume_init.detach().cpu().numpy().astype(np.float32).tofile(f"volume_init{index}.raw")
            #outputs.detach().cpu().numpy().astype(np.float32).tofile(f"out{index}.raw")
            #labels.detach().cpu().numpy().astype(np.float32).tofile(f"lab{index}.raw")
            volume = outputs.detach()
            origin = x_pred.detach()
            test_loss = loss.item()
            del outputs,x_pred
        endtime = time.time()
        duringtime = endtime - begintime
        print(duringtime)
        test_loss_sum+=test_loss
        result_vol_path = os.path.join(result_vol_dir,f"reconstructed_{index}.raw")
        # Saving the reconstructed results
        volume.detach().cpu().numpy().astype(np.float32).tofile(result_vol_path)
        lossfile_lists.append(losslist)

    test_loss_avg = test_loss_sum / len(test_loader)
    print(f"Average Test Loss: {test_loss_avg}")

# Optionally save the final state of the volume after testing
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
fig, ax = plt.subplots()

# 遍历每个列表并绘制在同一图表上
for i, lst in enumerate(lossfile_lists):
    ax.plot(range(1, len(lst) + 1), lst, color=colors[i], label=f'data {i+1}')

# 设置横坐标刻度为整数
ax.set_xticks(range(1, max(len(lst) for lst in lossfile_lists) + 1))

# 添加图例
plt.legend()


# 设置横坐标刻度为整数


plt.xlabel('Iteration times')
plt.ylabel('Value')
plt.title('Loss')
# 显示图像
plt.show()
# begin draw pic

# 保存图像
plt.savefig('loss.png')
