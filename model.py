import torch
from torch import nn
from dataset import CrowdDataset

#
class MCNN(nn.Module):
    def __init__(self, load_weights=False):
        super(MCNN, self).__init__()
        # 根据网络结构实现代码
        self.branch1 = nn.Sequential(
            # 计算输出特征图大小：(M+4*2-9)/1+1=M,即保持尺寸不变
            nn.Conv2d(3, 16, 9, padding=4),
            # inplace 将计算得值覆盖原来的值
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 7, padding=3),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 20, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 40, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(40, 20, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(20, 10, 5, padding=2),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(3, 24, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 24, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 12, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(30, 1, 1)
        )
        if not load_weights:
            self.initialize_weights()

    def forward(self, img_tensor):
        x1 = self.branch1(img_tensor)
        x2 = self.branch2(img_tensor)
        x3 = self.branch3(img_tensor)
        x = torch.cat((x1, x2, x3), 1)
        x = self.fuse(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# test code
if __name__ == '__main__':
    model = MCNN()
    img_root = r'D:\project\py_prj\MCNN-pytorch\Dataset\ShanghaiTech\part_A\train_data\images'
    gt_map_root = img_root.replace('images', 'ground_truth')

    dataset = CrowdDataset(img_root, gt_map_root, gt_down_samples=1)

    print('length of dataset:', len(dataset), '\n')
    img_tensor, gt_map_tensor = dataset[0]
    print('img_tensor:', img_tensor.shape, '\n')
    output = model(img_tensor.permute(2, 0, 1).unsqueeze(0))
    print(output.shape)
