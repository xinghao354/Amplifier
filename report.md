# 一、论文总结
## 1.1 研究遇到的难题

在时间序列预测中，现有深度学习模型（如Transformer、MLP、TCN等）存在一个普遍问题，它们倾向于学习高能量成分，而忽视低能量分量（low-energy components），即频谱中振幅较小的频率成分。这些分量虽能量低，但可能包含关键信息（如气象微变、金融微波动）。这导致模型无法充分利用时间序列中的所有信息，从而影响预测精度。实验表明，直接剔除低能量分量会显著增加预测误差（MSE↑），且模型训练时因高能量分量主导损失函数，导致低能量分量参数更新效率低下，进一步被“忽略”。

## 1.2 研究难题解决方法（创新点）

为了解决上述问题，作者提出了能量放大技术，并基于此构建了一个新的时间序列预测模型——Amplifier。

能量放大技术：

该技术包括两个核心模块，能量放大块和能量恢复块。

能量放大块：通过频谱翻转将高能量区域的能量“转移”到低能量区域，使低能量成分的能量增强，生成双能量峰频谱。

公式如下：

<img width="431" height="39" alt="image" src="https://github.com/user-attachments/assets/63be545d-def8-45de-9e2d-b5a4dabbdc46" />

##
<img width="434" height="66" alt="image" src="https://github.com/user-attachments/assets/f0d97ac8-1569-489a-9cf0-54d1e7e4855f" />

## 
<img width="441" height="36" alt="image" src="https://github.com/user-attachments/assets/667539e7-b13d-457f-981a-28c7b50adeb9" />
 
目的是让模型在训练中平等对待高、低能量成分。

能量恢复块：

通过频域线性操作（公式 Y′ = X′W + B ）预测需移除的翻转频谱，再从放大后的频谱中减去该分量，最终逆变换回时域，恢复原始能量水平。公式如下。

##
<img width="398" height="34" alt="image" src="https://github.com/user-attachments/assets/0d4f0156-92bc-42c0-b1ad-261e5cf73ab3" />

##
<img width="424" height="121" alt="image" src="https://github.com/user-attachments/assets/5f8d1247-9d89-492c-b52c-f2826d0a28ae" />


## 1.3、论文方法流程图

<img width="390" height="586" alt="image" src="https://github.com/user-attachments/assets/9dbf221b-d8ff-498c-b08c-8de1a14e0850" />

<img width="354" height="1015" alt="image" src="https://github.com/user-attachments/assets/a124b7f8-09e1-47ce-b421-52e9a9e2ca5f" />


## 1.4、论文结论

对现有方法常忽视低能量分量的问题，本文提出由能量放大模块与能量恢复模块构成的能量放大技术。该技术的核心思想是通过频谱翻转放大低能量分量的能量值，从而增强模型对低能量分量的关注度与处理能力。为充分发挥能量放大技术优势，我们设计了一种新型时间序列预测模型Amplifier。在八个真实数据集上的综合实验验证了所提模型的优越性。


# 二、论文公式和程序代码文件名行数对照表

## 2.1、论文模型所用公式及其对应代码表格如下

<img width="845" height="620" alt="image" src="https://github.com/user-attachments/assets/580ec158-a52e-4c90-95a0-d6bbec79b743" />

# 三、安装说明

源码链接：https://github.com/aikunyi/Amplifier?utm_source=catalyzex.com

## 3.1、创建环境

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


# 五、论文公式对应代码注释

能量放大器模块

<img width="860" height="147" alt="image" src="https://github.com/user-attachments/assets/40f72ff8-ba83-426d-9c40-362fd6452a51" />

SCI模块

<img width="1047" height="402" alt="image" src="https://github.com/user-attachments/assets/fae578cd-efa4-496b-a926-7cb42ee48c40" />

季节-趋势预测器模块

<img width="1076" height="161" alt="image" src="https://github.com/user-attachments/assets/1cae29a5-bf02-40fb-819e-997dab80c64c" />








