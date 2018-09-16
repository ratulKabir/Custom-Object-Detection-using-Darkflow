## Intro

While learning YOLO I have gone through a lot of blogs, github codes, blogs, courses. I have tried to combine all of them and see how to work with my own dataset set.<br>

I have used Anaconda and jupyter notebook. Here I have used Darkflow to detect custom object.<br>
Also I use Windows. Therefore all my tips are likely to run well on Windows. 

## Requirements

Python3, tensorflow 1.0, numpy, opencv 3. Links for installation below:


- [Python 3.5 or 3.6, Anaconda](https://www.youtube.com/watch?v=T8wK5loXkXg)
- [Tensorflow](https://www.youtube.com/watch?v=RplXYjxgZbw&t=91s). I recommend using the tensorflow GPU version. But if you don't have GPU, just go ahead and install the CPU versoin.<br>GPUs are more than 100x faster for training and testing neural networks than a CPU. Find more [here](https://pjreddie.com/darknet/hardware-guide/)
- [Opencv](https://anaconda.org/conda-forge/opencv)
## Download the Darkflow repo

- Click [this](https://github.com/thtrieu/darkflow)
- Download and extract the files somewhere locally


## Getting started

You can choose _one_ of the following three ways to get started with darkflow. If you are using Python 3 on windows you will need to install Microsoft Visual C++ 14.0. [Here](https://www.scivision.co/python-windows-visual-c++-14-required/) you can find installation process, why it is required, references etc or you can try [stack<b>overflow</b>](https://stackoverflow.com/).

1. Just build the Cython extensions in place. NOTE: If installing this way you will have to use `./flow` in the cloned darkflow directory instead of `flow` as darkflow is not installed globally.
    ```
    python3 setup.py build_ext --inplace
    ```

2. Let pip install darkflow globally in dev mode (still globally accessible, but changes to the code immediately take effect)
    ```
    pip install -e .
    ```

3. Install with pip globally
    ```
    pip install .
    ```

## Download a weights file

- Download the YOLOv2 608x608 weights file here (https://pjreddie.com/darknet/yolov2/)
- Read more about YOLO (in darknet) and download weight files [here](http://pjreddie.com/darknet/yolo/). In case the weight file cannot be found, you can check [here](https://drive.google.com/drive/folders/0B1tW_VtY7onidEwyQ2FtQVplWEU), which include `yolo-full` and `yolo-tiny` of v1.0, `tiny-yolo-v1.1` of v1.1 and `yolo`, `tiny-yolo-voc` of v2. Owner of this weights is [Trieu](https://github.com/thtrieu).
- NOTE: there are other weights files you can try if you like
- create a ```wights``` folder within the ```darkflow-master``` folder
- put the weights file in the ```weights``` folder

## Make own Dataset

I have spent one day to take photos, resize them and annotate them. I managed around 250 images. I recommend to have a much bigger dataset for better performance.<br>
#### Dataset
To make a dataset of objects around you 
- start taking photos of the objects that you want to detect.
- make sure have pictures from different angles, different poses, in different environment etc. 
- try to make the dataset as big as possible for better performance.<br> 

#### Annotation
- To annotate images download [labelImg](https://tzutalin.github.io/labelImg/). 
- Check this [video](https://www.youtube.com/watch?v=p0nR2YsCY_U&feature=youtu.be) to learn how to use lebelImg.<br>
- Github repo for labelImg can be found [here](https://github.com/tzutalin/labelImg#installation)

### Training on your own dataset

*The steps below assume we want to use tiny YOLO and our dataset has 3 classes*

1. Create a copy of the configuration file `tiny-yolo-voc.cfg` and rename it according to your preference `tiny-yolo-voc-3c.cfg` (It is crucial that you leave the original `tiny-yolo-voc.cfg` file unchanged, see below for explanation). Here `tiny-yolo-voc-3c.cfg` is for 3 classes, you can change the name as you wish.<br>

2. In `tiny-yolo-voc-3c.cfg`, change classes in the [region] layer (the last layer) to the number of classes you are going to train for. In our case, classes are set to 3.
    
    ```python
    ...

    [region]
    anchors = 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52
    bias_match=1
    classes=3  ## 3 classes
    coords=4
    num=5
    softmax=1
    
    ...
    ```

3. In `tiny-yolo-voc-3c.cfg`, change filters in the [convolutional] layer (the second to last layer) to num * (classes + 5). In our case, num is 5 and classes are 3 so 5 * (3 + 5) = 40 therefore filters are set to 40.
    
    ```python
    ...

    [convolutional]
    size=1
    stride=1
    pad=1
    filters=40  ## 5 * (3 + 5) = 40
    activation=linear

    [region]
    anchors = 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52
    
    ...
    ```

4. Change `labels.txt` to include the label(s) you want to train on (number of labels should be the same as the number of classes you set in `tiny-yolo-voc-3c.cfg` file). In my case, `labels.txt` will contain 3 labels.

    ```
    king
    ace
    ten
    ```
5. Reference the `tiny-yolo-voc-3c.cfg` model when you train.

    `python flow --model cfg/tiny-yolo-voc-3c.cfg --load weights/tiny-yolo-voc.weights --train --annotation train/Annotations --dataset train/Images --gpu 1.0 --epochs 300`
<br><br>In windows you need to type `python` at the beginning otherwise it does not recognise the flow command. Next spesify the model `--model cfg/tiny-yolo-voc-3c.cfg` and the weights `--load weights/tiny-yolo-voc.weights`. After that specify the path for the annatations `--annotation train/Annotations` and images `--dataset train/Images`. Use `--gpu 1.0` speed, if you have GPU just don't use this part. You can specify the number of epochs. By default it is 1000. However it can be stopped anytime. I recommend to keep the lose below 1. 
<br><br>

- Why should I leave the original `tiny-yolo-voc.cfg` file unchanged?
    
    When darkflow sees you are loading `tiny-yolo-voc.weights` it will look for `tiny-yolo-voc.cfg` in your cfg/ folder and compare that configuration file to the new one you have set with `--model cfg/tiny-yolo-voc-3c.cfg`. In this case, every layer will have the same exact number of weights except for the last two, so it will load the weights into all layers up to the last two because they now contain different number of weights.



## Object Detection using YOLO

Open the object-detection-with-YOLO.ipynb file. I have tried to add comments to make it easy to understand.

#### Image

To detect object from images:
1. Go to  the <b>Object Detection from Image</b> section. 
2. Change the image name with your image name from the following line
<br>
`img = cv2.imread('images/img_2386.jpg', cv2.IMREAD_COLOR)`
3. If you have multiple object in your image then you have to define all the ```tl```(Top left), ```br```(Bottom right) for different ofjects and their labels.
<br><br>My result<br><br>

I have put the image below to see if it detecs accurately<br>
<img src="img_2386.jpg" alt="Smiley face" height="242" width="242"><br>
It detects ace.
<br>
<img src="ace.png" alt="Smiley face" height="242" width="242">
#### Video

To detect object from video:
1. Go to  the <b>Object Detection from Video</b> section. 
2. Change the image name with your image name from the following line<br>
`capture = cv2.VideoCapture('test2.mkv')`
3. Run.
4. Press `Q` to quit

#### Webcam

To detect object from webcam just run the code from <b>Object Detection from Webcam</b> section. If you have multiple webcams you may need to specify the number correctly for your desired webcam. I have my laptops default webcam. Thats why I have used 0. To change the nummber edit this line<br>
`capture = cv2.VideoCapture(0)`
- Press `Q` to quit
<br><br>
My result<br><br>
<video controls src="YOLO.webm"/>


## References

- Real-time object detection and classification. Paper: [version 1](https://arxiv.org/pdf/1506.02640.pdf), [version 2](https://arxiv.org/pdf/1612.08242.pdf). 
- Official [YOLO](https://pjreddie.com/darknet/yolo/) website.

- I have learned YOLO, how it works from [coursera](https://www.coursera.org/lecture/convolutional-neural-networks/yolo-algorithm-fF3O0). Also Siraj has a nice [tutorial](https://www.youtube.com/watch?v=4eIBisqx9_g&t=1170s) on it. 

- The original darkflow repo is [this](https://github.com/thtrieu/darkflow) by [Trieu](https://github.com/thtrieu).

- To have video description of the codes and more understanding follow [this](https://www.youtube.com/watch?v=PyjBd7IDYZs&index=1&list=PLX-LrBk6h3wSGvuTnxB2Kj358XfctL4BM) videos. I have followed Mark Jay a lot whil making this project.
<br>
