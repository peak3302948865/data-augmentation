import mmcv
import numpy as np
from mmdet.apis import init_detector, inference_detector
import os
import matplotlib.pyplot as plt
from mmcv.visualization.image import imshow_det_bboxes

# 配置文件路径
yolov3_config_file = r"/hy-tmp/mmdetection/configs/yolo/yolov3_d53_8xb8-320-273e_coco_copy.py"

# 训练好的模型权重文件路径
yolov3_checkpoint_file = r"/hy-tmp/mmdetection/work_dirs/yolov3_d53_8xb8-320-273e_coco_copy/epoch_273.pth"

# 初始化检测模型
yolov3_model = init_detector(yolov3_config_file, yolov3_checkpoint_file)#, device='cuda:0')

# VOC数据集类别
class_names = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# 图片文件夹路径
img_folder = r"/hy-tmp/mmdetection/work_dirs/test_minedata/"
# 结果保存路径
output_folder = r"/hy-tmp/mmdetection/work_dirs/predict_test_minedata/"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取文件夹中的所有图片文件
img_files = [f for f in os.listdir(img_folder) if f.endswith('.jpg') or f.endswith('.png')]
# img_files = ['car bus person.jpg']
print(img_files)
# 对每张图片进行推理并保存对比结果
for img_file in img_files:
    img_path = os.path.join(img_folder, img_file)
    print(img_path)
    
    # YOLOv3 推理
    yolov3_result = inference_detector(yolov3_model, img_path)
    yolov3_bboxes = yolov3_result.pred_instances.bboxes.cpu().numpy()
    yolov3_labels = yolov3_result.pred_instances.labels.cpu().numpy()
    yolov3_scores = yolov3_result.pred_instances.scores.cpu().numpy()
    yolov3_bboxes_with_scores = np.hstack([yolov3_bboxes, yolov3_scores[:, np.newaxis]])

    # 读取图像
    img = mmcv.imread(img_path)

    # 可视化并保存结果
    yolov3_img = imshow_det_bboxes(
        img.copy(),
        yolov3_bboxes_with_scores,
        yolov3_labels,
        class_names=class_names,
        score_thr=0.3,
        show=False
    )
    
    plt.figure(figsize=(10,12))
    plt.imshow(yolov3_img)
    plt.title("YoloV3")
    plt.axis("off")
    
    # 保存图
    output_path = os.path.join(output_folder, img_file)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
