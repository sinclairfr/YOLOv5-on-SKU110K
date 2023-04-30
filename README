YOLOv5-SKU110k-Detection
========================

This repository contains the code, dataset, and training details for a custom object detection model using YOLOv5 trained on the SKU110k dataset. The model aims to detect and classify retail products in images, making it suitable for various applications such as inventory management, retail analytics, and smart shopping.

Table of Contents
-----------------

-   [Introduction](#introduction)
-   [Installation](#installation)
-   [Dataset](#dataset)
-   [Training](#training)
-   [Evaluation](#evaluation)
-   [Inference](#inference)
-   [License](#license)
-   [Acknowledgements](#acknowledgements)

Introduction
------------

YOLOv5 is a state-of-the-art object detection model known for its real-time performance and high accuracy. In this repository, we train YOLOv5 on the SKU110k dataset, which contains images of retail products in various environments.
Installation
------------

1.  Clone the repository:

    ```
    git clone https://github.com/yourusername/YOLOv5-SKU110k-Detection.git
    cd YOLOv5-SKU110k-Detection
    ```


2.  Install dependencies:
   
    ```
    ip install -r requirements.txt
    ```

3.  Download the pre-trained weights (optional):

    ```
    wget https://link-to-pretrained-weights/yolov5s.pt
    ```

Dataset
-------

The SKU110k dataset is a large-scale dataset containing 11,762 images with 1,294,594 annotated bounding boxes, each corresponding to a retail product. The dataset is publicly available and can be downloaded from [this link](http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz).

1.  Download and extract the dataset:

    ```
    wget http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz
    tar -xvf SKU110K_fixed.tar.gz
    ```

2.  Prepare the dataset for YOLOv5 by running the provided script:

    ```
    python prepare_dataset.py --input_path SKU110K_fixed --output_path SKU110k_yolov5
    ```

Training
--------

To train YOLOv5 on the SKU110k dataset, run the following command:

    ```
    python train.py --img 640 --batch 16 --epochs 100 --data sku110k.yaml --cfg yolov5s.yaml --weights yolov5s.pt
    ```

Replace `yolov5s.yaml` and `yolov5s.pt` with other YOLOv5 variants (such as `yolov5m`, `yolov5l`, or `yolov5x`) for larger models and improved performance.

Evaluation
----------

To evaluate the trained model on the validation set, run the following command:


    ```
    python val.py --data sku110k.yaml --weights runs/train/exp/weights/best.pt --iou-thres 0.5
    ```

Inference
---------

To run inference on sample images or a video, use the `detect.py` script:

    ```
    python detect.py --source path/to/your/image_or_video --weights runs/train/exp/weights/best.pt --conf 0.25 --iou-thres 0.5
    ```

The output images or video with bounding boxes will be saved in the `runs/detect` folder.

License
-------

This project is licensed under the [MIT License](https://www.wikiwand.com/fr/Licence_MIT).

Acknowledgements
----------------

-   [YOLOv5](https://github.com/ultralytics/yolov5) by Ultralytics
