import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import MCNN
from dataset import CrowdDataset
from torch.utils.data import DataLoader
#
# 使用gpu加速训练
device = torch.device("cuda")

# 数据集路径
img_root = r'Dataset\ShanghaiTech\part_A\train_data\images'
gt_root = r'Dataset\ShanghaiTech\part_A\train_data\ground_truth'
test_img_root = r'Dataset\ShanghaiTech\part_A\test_data\images'
test_gt_root = r'Dataset\ShanghaiTech\part_A\test_data\ground_truth'

# 加载数据集
dataset = CrowdDataset(img_root, gt_root, 4)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
test_dataset = CrowdDataset(test_img_root, test_gt_root, 4)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 训练集和测试集的长度
train_len = len(dataloader)
test_len = len(test_dataset)

# 创建网络模型
mcnn = MCNN().to(device)

# 定义损失函数
loss_fn = torch.nn.MSELoss().to(device)

# 定义优化器
learning_rate = 1e-6
optimizer = torch.optim.SGD(mcnn.parameters(), lr=learning_rate, momentum=0.95)

# 设置训练参数
epochs = 100
total_train_step = 0
total_test_step = 0
best_epoch = -1
best_mae = float('inf')

# 判断是否存在train_log文件夹，不存在则创建
if not os.path.exists('./train_log'):
    os.mkdir('./train_log')

# 判断是否存在checkpoints文件夹，不存在则创建
if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')

# 创建tensorboard对象
writer = SummaryWriter('./train_log')

for epoch in range(epochs):
    # 训练模式
    mcnn.train()
    epoch_loss = 0
    bar = tqdm(enumerate(dataloader), desc=f'Epoch [{epoch + 1}/{epochs}]', unit='image', total=train_len)
    for i, (img, gt_map) in bar:
        img_ = img.to(device)
        gt_map_ = gt_map.to(device)
        es_map = mcnn(img_)
        loss = loss_fn(es_map, gt_map_)
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新进度条
        bar.set_postfix(epoch_loss=epoch_loss)
        bar.update(1)
    writer.add_scalar('train_loss', epoch_loss, epoch)

    # 测试模式
    mcnn.eval()
    mae = 0
    for i, (img, gt_map) in enumerate(test_dataloader):
        img_ = img.to(device)
        gt_map_ = gt_map.to(device)
        es_map = mcnn(img_)
        mae += abs(es_map.detach().sum() - gt_map_.detach().sum()).item()

    mae /= test_len
    writer.add_scalar('test_mae', mae, epoch)
    print(f'Epoch: {epoch + 1}/{epochs}, MAE: {mae}')

    if mae < best_mae:
        best_mae = mae
        best_epoch = epoch
        torch.save(mcnn.state_dict(), './checkpoints/best_model.pth')

print(f'best_mae: {best_mae}, best_epoch: {best_epoch}')
writer.close()




