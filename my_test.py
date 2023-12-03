import visdom
import numpy as np

# 创建 Visdom 对象
vis = visdom.Visdom()

# 示例：可视化损失曲线
x = np.arange(1, 11)
y = np.random.rand(10)

# 绘制曲线
vis.line(X=x, Y=y, opts=dict(title='Loss Curve', xlabel='Epoch', ylabel='Loss'))
#