以 Pose3d 为例，这些是我多年配环境经验的总结。

## Step 1. CUDA

### 理解 **NVIDIA Driver 和 CUDA Toolkit**

- 显卡： 简单理解这个就是我们前面说的GPU，尤其指NVIDIA公司生产的GPU系列，因为后面介绍的cuda,cudnn都是NVIDIA公司针对自身的GPU独家设计的。
- 显卡驱动：很明显就是字面意思，通常指**NVIDIA Driver**，其实它就是一个驱动软件，而前面的显卡就是硬件。
- **CUDA** 是一个并行计算平台和编程模型，能够使得使用GPU进行通用计算变得简单和优雅
- **CUDA Toolkit** 是一个 CUDA 工具箱，包含了所有几乎深度学习需要的所有 cuda 组件
- `nvcc`其实就是CUDA的编译器,可以从CUDA Toolkit的`/bin`目录中获取,类似于`gcc`就是c语言的编译器。

在命令行输入nvidia-msi 在右上角能够看到 cuda version，这里指的是**NVIDIA Driver 的版本。**

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725932298701-4f30c562-63b2-40ae-9c13-a38a81d84021.png)

- **CUDA Driver**: 运行CUDA应用程序需要系统至少有一个**具有CUDA功能的GPU**和**与CUDA工具包兼容的驱动程序**。**每个版本的CUDA工具包都对应一个最低版本的CUDA Driver**，也就是说如果你安装的CUDA Driver版本比官方推荐的还低，那么很可能会无法正常运行。通常为了方便，在安装CUDA Toolkit的时候会默认安装CUDA Driver。
- **CUDA Driver是向后兼容的。这意味着根据CUDA的特定版本编译的应用程序将继续在后续发布的Driver上也能继续工作**。也就是说 **Driver 可以很新，因为它是向后兼容的。**
- 我们所说的 cuda 版本不是 driver 版本，一般指的是 **CUDA Toolkit 的版本。**

命令行输入nvcc 能够看到 的版本，这里指的是**CUDA Toolkit 的版本。**

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725932320182-58399942-809b-4dcd-9c19-8c11b158e563.png)

我们下面提到的 cuda 版本都是指**CUDA Toolkit 的版本**

### NVIDIA Driver 和 CUDA Toolkit 的安装

#### 如何选择合适的 cuda 版本

我一般先定 pytorch 版本，根据 pytorch 版本确定 cuda 版本。

在项目的 README.md 中一般会标明，如果标明了 cuda 版本，直接选择它的即可，如果标明了 torch 版本，我们则需要到[https://download.pytorch.org/whl/torch_stable.html](https://download.pytorch.org/whl/torch_stable.html)寻找对应版本的 torch。

例如，这个项目要求的 pytorch 版本为 1.7.1

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725933123173-a200e2b1-528b-4ae3-8a92-d21eea2f8b5c.png)

在[https://download.pytorch.org/whl/torch_stable.html](https://download.pytorch.org/whl/torch_stable.html)中搜索 torch-1.7.1，可以看到前缀有 cpu,cu110 等，我们选择 cu110 即 cuda11.0 版本的 torch，这样，我们就安装 11.0 的 cuda 就好了。

如果前面的前缀已经有你目前的 cuda 版本，那么你直接跳到 Step2 安装 torch 即可。

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725933215813-f4f63a41-9fa1-4c64-a368-150ce234f6e2.png)

很多时候，我们需要安装适应 pytorch 版本的 cuda，我一般选择在电脑上配置多个 cuda 备用。

在[https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)可以找到所有的 cuda 版本。

#### Linux 安装

找到需要的 cuda 版本后，选择自己的操作系统，CPU 架构，系统发行版本，安装类型选择 runfile(local)

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725933619644-1e2d052c-df2c-495a-8e9a-26e8a454bca4.png)然后在 linux 终端运行下面Installation Instructions中的指令

下载安装包

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725933866402-c52691b8-e302-40a9-96c6-c39f60720c82.png)

运行.run 文件，输入 accept

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725934046712-863f9f3d-972c-4180-9628-435879cab5bf.png)

下面是选择界面：

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725934069595-7bf5d723-9b1d-479d-9411-2d8c1d9b31c2.png)

一般来说，如果您的 **NVIDIA Driver 中的（即运行** nvidia-msi 在右上角能够看到的）cuda version 高于现在要安装的 cuda 版本，则可以不用勾选 Driver，只需要勾选 Cuda Toolkit 即可，反之则需要勾选Driver。余下的几个勾不勾选影响不大。

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725934217600-1d80bcd1-d1ea-4cb8-94e3-5a46a0c66cef.png)

选择 Intsall 等待安装完成即可。

之后重启终端，运行 nvcc -V 验证当前 cuda 版本。

#### Windows 安装

选择自己的操作系统，CPU 架构，系统版本，下载 exe（local）然后运行。

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725934368244-e3713935-1e1a-4971-ab50-b0033e2ba7b1.png)

这里选择自己合适的路径即可，一般不用改变。

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725934679627-03f54854-46bb-497a-9613-b0becba4c388.png)

选择自定义。

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725935152078-05f0770a-622f-473e-9dbe-a5de5aa8e2c8.png)

同样的，如果您的 **NVIDIA Driver 中的（即运行** nvidia-msi 在右上角能够看到的）cuda version 高于现在要安装的 cuda 版本，则可以不用勾选 Driver，只需要勾选 Cuda 即可，反之则需要勾选Driver components。

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725935178943-8eed6028-2df2-483b-9811-7f1f1a970d00.png)

之后按照指引安装完成即可。

## Step 2. Ananconda 安装和虚拟环境搭建

如果你已经安装好了Ananconda，则可以跳过这一步。

### Windows 安装

如果没有，打开[https://www.anaconda.com/download](https://www.anaconda.com/download)，可以填邮件注册也可以选择跳过注册。

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725934756461-ad67f4cc-72a7-463b-8c95-b80da52d1b39.png)

选择你的平台和 cpu 架构。

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725934840273-431e925a-cb6a-4f5c-8e76-4abdafc863cd.png)

直接找到刚才下载好的文件双击打开。如图

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725935788653-545d913b-3546-4e81-ba78-9179fa0ac90b.png)

这里选择All Users

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725935788795-6c1788c9-59d0-4478-bece-87e050adbcb0.png)

下一步，选择安装路径，最好不要有中文和空格

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725935788705-11b24a8d-019c-444f-acf2-53329f8f5bb7.png)

我是在E盘建立了一个新的文件夹，E:\Anaconda，注意这个文件夹不要使用中文或者空格、特殊字符。选择对应安装目录，点next。![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725935788532-a6e6b5f8-2a27-467e-82c9-1718ad08c9f2.png)

第一个选项意思就是将安装路径填入到系统环境变量中，这里勾选，后面使用着会出现问题，一般还是不要，然后手动添加环境变量。

第二个勾选默认的不用管。直接点击 Install

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725935789241-50bc6900-b132-4627-83d6-80cc40fe3b1e.png)

安装完成有提示

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725935789404-0473041d-8a9d-48b3-823c-93724ad85e33.png)接下来开始配置环境变量

这是步骤：此电脑----->属性----->高级系统设置----->环境变量----->path----->编辑----->新建(好多软件都是这里配置环境变量，大家应该不陌生)，懒得话直接按win键，搜索“环境变量”

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725935789569-93897ae0-3ee5-479c-824b-84064149f383.png)

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725935789631-cce85d17-ffc5-4d59-98ca-e6d6920ee206.png)

双击Path,点击新建

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725935789902-ddbc5ad8-199a-4ae3-900e-e7c6efb47cb7.png)

把这几条复制到里面（注意我装的是E:\Anaconda这个目录，你要根据自己实际安装目录进行改动，比如装在C盘根目录，就把所有E改成C就行，）：

```
E:\Anaconda
E:\Anaconda\Scripts
E:\Anaconda\Library\mingw-w64\bin
E:\Anaconda\Library\usr\bin
E:\Anaconda\Library\bin
```

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725935790078-a46a84de-2981-4c9f-bc84-0cf12cc52329.png)

测试是否配置成功，进入cmd:

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725935790238-9f4cb01b-5819-4954-a4c2-b75f92325d90.png)

输入 python,看是否有python环境。

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725935790488-42d87cc7-3b22-4136-b9f8-1276dbd6ae55.png)

然后在cmd中输入 conda --version ，如图就是有conda环境

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725935790743-57b677a6-1c3e-44b2-94f8-6f27e4117fae.png)

如果提示conda不是内部或外部命令，那一般是，Anaconda的 环境变量 没配置好。好好检查一下。

### Linux 安装

在命令行输入如下指令

```
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
sh Anaconda3-2024.06-1-Linux-x86_64.sh
```

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725936177972-bd6e5311-0cff-4f7a-971e-ff7eafed8879.png)

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725936283197-0ff9e0ff-092c-4a00-b266-e460d20f48e3.png)

一点一点翻到底（按 enter），然后输入 accept 就可以。

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725936523823-34e52651-ccd2-42d6-9b06-2b81695dbd2a.png)

之后一路确认就行，它会自动帮你配置好当前 shell。

### 创建虚拟环境

找到项目要求的 Python版本，在 cmd 运行下面的指令（win 和 linux 都一样）其中 pose3d 是环境的名称。

```
conda create -n pose3d python=3.9
conda activate pose3d
```

在命令行前出现 pose3d 的环境名字，则说明配置成功。

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725937453135-345e0602-ebcd-428c-a27a-2dca05ae38b8.png)

**注意：以后所有的操作都要在这个环境下！**

**注意：以后所有的操作都要在这个环境下！**

这样可以不影响以后其他项目的环境，也可以方便打包。

## Step 3. Pytorch 和 Torchvision 安装

### Pytorch

[https://download.pytorch.org/whl/torch_stable.html](https://download.pytorch.org/whl/torch_stable.html)中可以找到所有的 Pytorch 版本，找到我们的 cuda 版本对应的，项目要求的 torch 版本：

注意 cuda 版本（刚才安装），Python版本（如 cp38 表示python=3.8 ，虚拟环境的）和系统架构必须与自己安装的完全对应。

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725936662604-92fb12fa-c5ff-4112-98d5-c22f29bcc3f0.png)

右键复制链接

然后运行，install 后面的是刚才复制的链接

```
pip install https://download.pytorch.org/whl/cu110/torch-1.7.1%2Bcu110-cp39-cp39-linux_x86_64.whl
```

### Torchvison

下面是 torch 和 torchvision 的版本对照表

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725936891666-38242ad0-8c06-4778-8e83-a149b9785a8e.png)

比如我们这次的 torch1.7.1 对应的就是 torchvision0.8。

只要是 0.8.x 都可以，但是在 README 中官方指定了 0.8.2 我们就安装 0.8.2

同样在[https://download.pytorch.org/whl/torch_stable.html](https://download.pytorch.org/whl/torch_stable.html)中搜索 cu111 的torchvision0.8 复制链接进行安装。注意 cuda 版本，Python版本和系统架构的对应。

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725937836092-96e86ef1-f3d3-4937-af8f-d3c7792a9262.png)

```
pip install https://download.pytorch.org/whl/cu110/torchvision-0.8.2%2Bcu110-cp39-cp39-linux_x86_64.whl
```

在命令行输入 python（注意在 pose3d 环境下）

输入 import torch 然后torch.cuda.is_available()如果返回 True 则说明安装成功

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725937968213-ef30d11a-673c-48d7-81de-17fd1581f59b.png)

## Step 4. mmcv 安装

这个项目需要安装 mmcv， 打开[https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-mmcv](https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-mmcv)

找到 Install with pip，在这里选择自己的平台，cuda 版本和 torch版本，官方 README 要求的是 mmcv>=2.0 这里也满足。

![](https://cdn.nlark.com/yuque/0/2024/png/32601160/1725937565491-338102f1-6045-491f-b596-4f6a11183ec5.png)

然后运行它给出的指令即可。

```
pip install mmcv==2.0.0rc4 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7/index.html
```

## Step 5. 其他依赖安装

在命令行依次运行以下指令

```
pip3 install -r requirements_new.txt
pip install -U openmim
mim install mmengine
mim install "mmpose>=1.1.0"
mim install mmdet
```

这个是根据这个项目官方的要求改变编的，requirements_new.txt 中加了一些原来requirements.txt 没有但需要的依赖。

如果前面的都正确的话，这个一般不会有什么报错，到此项目环境配置完成。
