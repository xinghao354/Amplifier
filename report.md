# 一、论文总结
## 1.1 研究遇到的难题

在时间序列预测中，现有深度学习模型（如Transformer、MLP、TCN等）存在一个普遍问题，它们倾向于学习高能量成分，而忽视低能量分量（low-energy components），即频谱中振幅较小的频率成分。这些分量虽能量低，但可能包含关键信息（如气象微变、金融微波动）。这导致模型无法充分利用时间序列中的所有信息，从而影响预测精度。实验表明，直接剔除低能量分量会显著增加预测误差（MSE↑），且模型训练时因高能量分量主导损失函数，导致低能量分量参数更新效率低下，进一步被“忽略”。

## 1.2 研究难题解决方法（创新点）

为了解决上述问题，作者提出了能量放大技术，并基于此构建了一个新的时间序列预测模型——Amplifier来解决低能量分量被忽视的问题，其包含两个模块。

##
能量放大技术：

该技术包括两个核心模块，能量放大块和能量恢复块。

能量放大块：通过频谱翻转将高能量区域的能量“转移”到低能量区域，使低能量成分的能量增强，生成双能量峰频谱，迫使后续模块平等学习。

公式如下：

频谱翻转：把整条复数频谱沿频率轴镜像使得低频、高频对调，其幅度不变，相位共轭（因为实数信号 DFT 共轭对称），最终原来能量低的高频区搬到了低频区，低能量分量瞬间变成高能量。

<img width="431" height="39" alt="image" src="https://github.com/user-attachments/assets/63be545d-def8-45de-9e2d-b5a4dabbdc46" />

能量放大：逐复数相加，原谱 + 翻转谱。由于翻转后低能量高频区被搬到低频区，所以相加后原低能量分量幅度平方翻倍 → 能量翻倍。后再逆变换回时域，得到能量放大后的信号 X_amp，长度仍为 L。

<img width="434" height="66" alt="image" src="https://github.com/user-attachments/assets/f0d97ac8-1569-489a-9cf0-54d1e7e4855f" />

能量计算公式

<img width="441" height="36" alt="image" src="https://github.com/user-attachments/assets/667539e7-b13d-457f-981a-28c7b50adeb9" />

 ##
能量恢复块：

通过频域线性操作（公式 Y′ = X′W + B ）预测需移除的翻转频谱，再从放大后的频谱中减去该分量，最终逆变换回时域，恢复原始能量水平。公式如下。

频域长度映射 + 加偏置

复数仿射变换：对每个频率做复数乘加，允许幅度缩放 + 相位旋转。W 和 B 在训练过程中会自动学会该减多少翻转谱才最合适。

<img width="398" height="34" alt="image" src="https://github.com/user-attachments/assets/0d4f0156-92bc-42c0-b1ad-261e5cf73ab3" />

##
能量恢复三部曲

一、先把时域预测信号转成频域，得到带有人工翻转谱的复数谱。

二、接着把前面“为了放大低能量而加进去的翻转谱”原样去除，得到干净、能量恢复后的频谱输出，其长度已与预测长度对齐。

三、最后逆变换回时域，得到最终预测值Ŷ，长度 τ，实数。能量水平已与原始信号一致，不会因放大操作而整体抬升幅值。

<img width="424" height="121" alt="image" src="https://github.com/user-attachments/assets/5f8d1247-9d89-492c-b52c-f2826d0a28ae" />


## 1.3、论文方法流程图

<img width="500" height="906" alt="image" src="https://github.com/user-attachments/assets/9dbf221b-d8ff-498c-b08c-8de1a14e0850" />

<img width="354" height="1015" alt="image" src="https://github.com/user-attachments/assets/a124b7f8-09e1-47ce-b421-52e9a9e2ca5f" />


## 1.4、论文结论

对现有方法常忽视低能量分量的问题，本文提出由能量放大模块与能量恢复模块构成的能量放大技术。该技术的核心思想是通过频谱翻转放大低能量分量的能量值，从而增强模型对低能量分量的关注度与处理能力。为充分发挥能量放大技术优势，我们设计了一种新型时间序列预测模型Amplifier。在八个真实数据集上的综合实验验证了所提模型的优越性。


# 二、论文公式和程序代码文件名行数对照表

## 2.1、论文模型所用公式及其对应代码表格如下

<img width="845" height="620" alt="image" src="https://github.com/user-attachments/assets/580ec158-a52e-4c90-95a0-d6bbec79b743" />

# 三、安装说明

源码链接：https://github.com/aikunyi/Amplifier?utm_source=catalyzex.com

## 3.1、数据准备

根据README文件中Data Preparation部分要求，前往谷歌网盘[Google Drive]下载数据，原网址为：(https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) 

下载压缩包解压后创建一个文件夹命名为dataset，将数据放入后将该文件夹与Amplifier-main文件夹放在一块。
数据按照文件要求放置例如"./dataset/ETT-small/ETTh1.csv"

## 3.2、创建环境

在conda里按照文件要求创建环境（命令如下）

conda create -n Amplifier python == 3.8.0

conda activate Amplifier

pip install -r requirements.txt

requirements.txt内容如下

numpy == 1.23.5

pandas == 1.5.3

scikit-learn == 1.2.2

torch == 2.0.1（需要卸载在官网找到合适的版本指令下载）

剩余的需求包在运行时，缺少什么 ， pip install自行下载后便可运行如下图

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/c2fea729-deb3-49e2-854d-fb214c333613" />

# 四、运行结果

打开项目后按照scripts文件夹里的文件调整参数训练，作者给出了ELC、ETTH1、ETTM1的训练参数。

# 五、论文公式对应代码注释

## 5.1 Amplifier.py模型-论文公式对应代码

能量放大器模块

公式5：

<img width="341" height="34" alt="image" src="https://github.com/user-attachments/assets/9e33d3dc-a3f4-4532-a1a7-770adbaee4af" />

公式6：

<img width="347" height="57" alt="image" src="https://github.com/user-attachments/assets/c2a0086a-85d7-4adb-9648-769fd0b9b940" />

代码注释：

```python
        # Energy Amplification Block（能量放大块）
        x_fft = torch.fft.rfft(x, dim=1)   # domain conversion 将时域输入 x 转成复数频谱（长度=seq_len//2+1）
        x_inverse_fft = torch.flip(x_fft, dims=[1])  # flip the spectrum  把频谱沿频率轴翻转，实现 X'[k] = X[T-k]
        x_inverse_fft = x_inverse_fft * self.mask_matrix # 用掩码屏蔽直流/奈奎斯特分量，防止能量泄漏（X'）
        x_amplifier_fft = x_fft + x_inverse_fft # 能量放大，公式：XAmp = X +X′
        x_amplifier = torch.fft.irfft(x_amplifier_fft, dim=1) # 回到时域，得到能量放大后的信号，即公式：XAmp = IDFT(XAmp)
```
SCI模块

公式10：

<img width="376" height="33" alt="image" src="https://github.com/user-attachments/assets/18953009-0940-4176-8be9-12e134bd2e6e" />

公式11：

<img width="340" height="64" alt="image" src="https://github.com/user-attachments/assets/9aa4a91d-f941-4a97-b2a7-2b6be5517bc9" />

代码注释：

```python
 # SCI block（半通道模块）
        if self.SCI:
            x = x_amplifier    # 用能量放大后的信号作为输入
            # extract common pattern
            # --- ① 提取共性 Common Pattern ---
            # 把 C 个变量压成 1 维，得到“抽象主通道”
            common_pattern = self.extract_common_pattern(x)   #共性提取,公式10：XCom = CompressionC(X)
            common_pattern = self.model_common_pattern(common_pattern.permute(0, 2, 1)).permute(0, 2, 1)
            # model specific pattern
            # --- ② 求特异性 Specific Pattern ---
            # 原信号剪掉共性 → 残差即各通道独有部分
            specififc_pattern = x - common_pattern.repeat(1, 1, C)   #特异性提取，即公式XSpc = X − XCp
            specififc_pattern = self.model_spacific_pattern(specififc_pattern.permute(0, 2, 1)).permute(0, 2, 1) # 对每条特异性序列单独做 FFN
                                                       #公式11：XSp = FFN(XSpc)
            # --- ③ 融合并输出 ---
            # 把建模后的特异性与共性加回，形成最终表示
            x = specififc_pattern + common_pattern.repeat(1, 1, C)
            x_amplifier = x                 # 覆盖变量，继续向下游传递
```

季节-趋势预测器模块

公式12：

<img width="373" height="28" alt="image" src="https://github.com/user-attachments/assets/2b8a23f0-8025-4489-82b4-c47c38fc3fdd" />

公式13：

<img width="396" height="61" alt="image" src="https://github.com/user-attachments/assets/955b04af-79c9-45f8-8c59-ee0d4eb76594" />

公式14：

<img width="356" height="40" alt="image" src="https://github.com/user-attachments/assets/f2771617-234e-491a-99c7-d0804ec7e31d" />

代码注释：

```python
        # Seasonal Trend Forecaster（季节-趋势预测器）
        # 把能量放大后的序列分解为季节项与趋势项
        seasonal, trend = self.decompsition(x_amplifier)    #季节趋势分解，即公式12：XSci Trend,XSci Season = STD(XSci)
        seasonal = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1) #季节预测，即公式13：YSci Season = Season-FFN(XSci Season)
        trend = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)  #趋势预测，即公式13：YSci Trend = Trend-FFN(XSci Trend)
        out_amplifier = seasonal + trend    #合并输出，即公式14：Y = YSci Trend +YSci Season
```

能量恢复模块

公式8：

<img width="329" height="32" alt="image" src="https://github.com/user-attachments/assets/8ebbe45e-5ad7-492b-8ef9-81440aa7548d" />

公式9：

<img width="353" height="103" alt="image" src="https://github.com/user-attachments/assets/f50a15f4-4294-475a-8140-0d7f82311474" />

代码注释：

```python
        # Energy Restoration Block（能量恢复模块）
        out_amplifier_fft = torch.fft.rfft(out_amplifier, dim=1)    #把季节-趋势预测器输出的时域信号转到频域，即能量恢复公式9：YAmp = DFT(YAmp)
        x_inverse_fft = self.freq_linear(x_inverse_fft.permute(0, 2, 1)).permute(0, 2, 1) #将早期保留的翻转谱长度映射到与预测谱一致（公式8：Y′=X′W+B）
        out_fft = out_amplifier_fft - x_inverse_fft                 #减掉人为加入的翻转谱，恢复原始能量水平，即能量恢复公式9：Y = YAmp −Y′
        out = torch.fft.irfft(out_fft, dim=1)                       #逆FFT回到时域，得到最终预测值Ŷ，即能量恢复公式9：Yˆ = IDFT(Y)
```

## 5.2 Amplifier.py模型其余部分代码大致注释

```python
import torch
import torch.nn as nn
from layers.RevIN import RevIN


# ------------------------------------------------------------------------------
# 滑动平均块：用简单平均提取“趋势”
# ------------------------------------------------------------------------------
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        #  一维平均池化 ≈ 滑动平均（stride=1 保证长度不变）
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        ## 两端镜像补零，防止边界被截断
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        ## 池化在时间维操作：先转置 → 池化 → 再转回来
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

# ------------------------------------------------------------------------------
# 季节-趋势分解：原始序列 = 季节残差 + 趋势
# ------------------------------------------------------------------------------
class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        # 滑动平均窗口大小由外部传入，stride 固定 1
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)       #趋势分量
        res = x - moving_mean                  #季节残差分量
        return res, moving_mean

# ------------------------------------------------------------------------------
# Amplifier 主模型
# ------------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # 读取参数
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.hidden_size = configs.hidden_size

        # >>> 可逆实例归一化（RevIN）用于稳定分布
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)

        # 季节-趋势分解窗口大小（25）
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)

        # 能量放大掩码：奈奎斯特分量（可训练）
        self.mask_matrix = nn.Parameter(torch.ones(int(self.seq_len / 2) + 1, self.channels))

        # 复数线性层：历史频谱长度 → 预测频谱长度
        self.freq_linear = nn.Linear(int(self.seq_len / 2) + 1, int(self.pred_len / 2) + 1).to(torch.cfloat)
        # >>> 季节分量预测器：T → pred_len
        self.linear_seasonal = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len)
        )

        # >>> 趋势分量预测器：同上结构
        self.linear_trend = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len)
        )

        # SCI block
        self.SCI = configs.SCI  # SCI 块开关
        # >>> 通道压缩：C → 1，抽“共性”
        self.extract_common_pattern = nn.Sequential(
            nn.Linear(self.channels, self.channels),
            nn.LeakyReLU(),
            nn.Linear(self.channels, 1)
        )

        # >>> 共性时序建模 FFN
        self.model_common_pattern = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.seq_len)
        )

        # >>> 特异性时序建模 FFN
        self.model_spacific_pattern = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.seq_len)
        )

    # ------------------------------------------------------------------------------
    # 前向传播
    # -----------------------------------------------------------------------------
    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, T, C = x.size()  # >>> 取维度：B=batch，T=历史长度，C=变量数

        # RevIN  // >>> 可逆归一化：减均值除标准差，稳定分布
        z = x
        z = self.revin_layer(z, 'norm')
        x = z

```
后面接5.1内容



