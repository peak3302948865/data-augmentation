### 3.1 安装MMDetection

step1. 使用 [MIM](https://github.com/open-mmlab/mim) 安装 [MMEngine](https://github.com/open-mmlab/mmengine) 和 [MMCV](https://github.com/open-mmlab/mmcv)

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

step2. 安装 MMDetection

方案 a：如果你开发并直接运行 mmdet，从源码安装它：

```shell
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
# "-v" 指详细说明，或更多的输出
# "-e" 表示在可编辑模式下安装项目，因此对代码所做的任何本地修改都会生效，从而无需重新安装。
```

方案 b：如果你将 mmdet 作为依赖或第三方 Python 包，使用 MIM 安装：

```shell
mim install mmdet
```

### 3.2 下载并准备VOC数据集

#### 3.2.1 下载并解压VOC数据集

```shell
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtrainval_06-Nov-2007.tar
```

#### 3.2.2 转化VOC数据集为COCO

运行文件`voc_to_coco.py`，将新生成的文件夹data/VOCdevkit/VOC2007_COCO移动到路径data/，并重命名文件夹为coco

### 3.3 修改yolov3的配置文件

#### 3.3.1 

拷贝原有的`configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco.py`，得到新文件`configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco_copy.py`，对其进行如下修改

1. 更改`num_classes=20`，`num_workers=8`

2. 最大训练周期max_epoch=273，每7epoch验证一次：`train_cfg = dict(max_epochs=273, val_interval=7) `

3. 更改文件读取的位置为`data/coco`，`ann_file='annotations/train2017.json'`

4. 优化器设置如下

   ```python
   optim_wrapper = dict(
       type='OptimWrapper',
       optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=5e-4),
       clip_grad=dict(max_norm=35, norm_type=2))
   ```

5. 学习率调整：在训练后期第 220 和第 250 个 epoch 时下降学习率

   ```python
   param_scheduler = [ 
       dict(type='MultiStepLR', by_epoch=True, milestones=[220, 250], gamma=0.1)
   ]
   ```

6. 加入如下代码：VOC数据集类别以及锚框颜色

   ```python
   metainfo = {
       'classes': (
           'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person',
           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
       ),
       'palette': (
           [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
           [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0],
           [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
           [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]
       )
   } 
   ```

#### 3.3.2

拷贝原有的`configs/yolo/yolov3_d53_8xb8-320-273e_coco.py`，得到新文件`configs/yolo/yolov3_d53_8xb8-320-273e_coco_copy.py`，在新文件进行如下修改：

```python
# 继承yolov3_d53_8xb8-ms-608-273e_coco_copy.py中的配置
_base_ = './yolov3_d53_8xb8-ms-608-273e_coco_copy.py'
# 载入从mmdetection官网下载的预训练的模型
load_from = r"/hy-tmp/mmdetection/configs/yolo/yolov3_d53_320_273e_coco-421362b6.pth"
```

### 3.4 开始训练

```shell
cd mmdetection
python tools/train.py configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco_copy.py
```

### 3.5 模型测试

训练完成后，进行测试。请注意确保模型文件存放的位置与下面代码所示相同。

```powershell
python tools/test.py configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco_copy.py work_dir/yolov3_d53_8xb8-320-273e_coco_copy/epoch_273.pth --out results_yolo.pkl
```

效果可视化：从测试集中取出识别效果最好的10张和最坏的10张

```
python tools/analysis_tools/analyze_results_copy.py configs/yolo/yolov3_d53_8xb8-320-273e_coco_copy.py results_yolo.pkl results --topk 10
```

### 3.6 目标检测任务实战

使用`test.py`脚本，使用训练好的模型在自己找的图片上进行推理。

请确保待检验图片的存放位置为`work_dir/test_minedata/`

```shell
python work_dirs/test.py
```

### 