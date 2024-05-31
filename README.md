### Mid-Term-Project
### 任务1：
微调在ImageNet上预训练的卷积神经网络实现鸟类识别
基本要求：
(1) 修改现有的CNN架构（如AlexNet，ResNet-18）用于鸟类识别，通过将其输出层大小设置为200以适应数据集中的类别数量，其余层使用在ImageNet上预训练得到的网络参数进行初始化；
(2) 在[CUB-200-2011]( https://data.caltech.edu/records/65de6-vp158)数据集上从零开始训练新的输出层，并对其余参数使用较小的学习率进行微调；
(3) 观察不同的超参数，如训练步数、学习率，及其不同组合带来的影响，并尽可能提升模型性能；
(4) 与仅使用CUB-200-2011数据集从随机初始化的网络参数开始训练得到的结果进行对比，观察预训练带来的提升。

### 任务2：
在VOC数据集上训练并测试目标检测模型Faster R-CNN和YOLO V3
基本要求：
（1） 学习使用现成的目标检测框架——如mmdetection或detectron2——在VOC数据集上训练并测试目标检测模型Faster R-CNN和YOLO V3；
（2） 挑选4张测试集中的图像，通过可视化对比训练好的Faster R-CNN第一阶段产生的proposal box和最终的预测结果。
（3） 搜集三张不在VOC数据集内包含有VOC中类别物体的图像，分别可视化并比较两个在VOC数据集上训练好的模型在这三张图片上的检测结果（展示bounding box、类别标签和得分）；

### 任务1请详见文件夹CNN_for_Birds_Identification, 任务2 Faster R-CNN部分详见文件夹 faster-r-cnn，YOLO V3部分详见文件夹 yolo-v3
