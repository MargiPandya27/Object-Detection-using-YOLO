# Object Detection using YOLOv1 

## RESULTS!!!
![image](https://user-images.githubusercontent.com/117746681/224552716-9ee4c0c1-9c6b-49e0-a7b5-dd2054a50623.png)
![image](https://user-images.githubusercontent.com/117746681/224552590-502db736-0412-4823-ad5b-c92555d9d848.png)
![image](https://user-images.githubusercontent.com/117746681/224552639-9addc03a-7649-4013-acce-545103c2e4dd.png)

**Before starting with the implementation what is object detection...**

## Object Detection
Objection Detection: The object detction algorithm tries to locate the presence of objects with a bounding box and types or classes of the located objects in an image.

How is object detection different from localization?

A object localization algorithm will output the presence of objects in an image and indicate their location with a bounding box whereas the object detction algorithm would also output the class or type of image.

## YOLO
YOLO model was first proposed by Joseph Redmon in 2015 by the title ***You Look Only Once**: Unified, Real-Time Object Detection* 

**Function**: The YOLO model takes image as input and will predicts bounding boxes and class labels for each bounding box directly.
Previous methods for this, like R-CNN and its variations, used a pipeline to perform this task in multiple steps. This can be slow to run and also hard to optimize, because each individual component must be trained separately. YOLO, does it all with a single neural network. 

**Algorithm**: 
The input image is divide into a SxS grid of cells. For each object that is present in the image will have center that would lie in one of the grids. This grid is responsible for predicting the predicting it.

The bounding box prediction will have 5 components: (x, y, w, h, confidence).
The (x, y) coordinate represents the centre of the box, relative to grid location. The box dimension are also normalized to [0, 1], relative to the image size.

![image](https://user-images.githubusercontent.com/117746681/224551361-b9cb402f-a072-49e0-81b5-874fee82d943.png)

**Confidence** reflects the presence or absence of an object of any class.
If object is present,
Confidence = Pr(Object) * IOU(pred, truth) 

If no object is not present,
confidence = 0

Prediction for bounding box in image will from each grid cell if makes B of those predictions, so there are in total S x S x B * 5 outputs related to bounding box predictions.
The output prediction will also consists of class probablities too  Pr(Class(i) | Object). This probability is conditioned on the grid cell containing one object. The network only predicts one set of class probabilities per cell, regardless of the number of boxes B. That makes S x S x C class probabilities in total

Adding the class predictions to the output vector, a **S x S x (B * 5 + C)** tensor is formed as output.

**Architecture**
![image](https://user-images.githubusercontent.com/117746681/224551462-8b1cd046-dd9c-483b-92b6-028ba2ee7490.png)
Architecture from Paper
![image](https://user-images.githubusercontent.com/117746681/224551429-4df08f90-363e-4656-84b5-6f66fa74c65f.png)
From this implementation, batch normalization is included to speed up the traing process and make the model less prone to overfitting.

**Loss Function**
![image](https://user-images.githubusercontent.com/117746681/224551559-bbc92048-c2c6-47c5-9907-2eb09246842c.png)

### YOLO Loss Function — Part 1 (Loss Function related to bounding predictions)
![image](https://user-images.githubusercontent.com/117746681/224551598-fe8cf46a-a2e7-4550-a09e-51ad50cc4aa3.png)

This equation computes the loss related to the predicted bounding box position (x,y). The function computes a sum over each bounding box predictor (j = 0.. B) of each grid cell (i = 0 .. S^2). 𝟙 obj is defined as follows:
1, If an object is present in grid cell i and the _j_th bounding box predictor is “responsible” for that prediction
0, otherwise

YOLO will predict multiple bounding boxes for a object. Only one predictor is assigned to be "responsible" it would be one with highest current IOU score.

    (x, y): predicted bounding box position,
    (x̂, ŷ): actual position from the training data.

### YOLO Loss Function — Part 2 (Loss Function related to dimensions)
![image](https://user-images.githubusercontent.com/117746681/224551618-9755f2a7-845d-4e62-8967-f7a22788a6c5.png)

This is the loss related to the predicted box width / height. The equation looks similar to the first one, except for the square root. What’s up with that? Quoting the paper again:

Our error metric should reflect that small deviations in large boxes matter less than in small boxes. To partially address this, the square root of the bounding box width and height are predicted instead of the width and height directly.

### YOLO Loss Function — Part 3 (Loss Function related to confidence score)
![image](https://user-images.githubusercontent.com/117746681/224551654-4a38bd90-8f9a-47e5-85a1-92f7eb21a7ed.png)

Here, the loss associated with the confidence score for each bounding box predictor is computed. 

    C: confidence score 
    Ĉ: is the intersection over union of the predicted bounding box with the ground truth.
    𝟙 obj: 1 when there is an object in the cell, 
           0 otherwise. 
    𝟙 noobj is the opposite.

λ parameters that are used to differently weight parts of the loss functions. This is necessary to increase model stability. The highest penalty is for coordinate predictions (λ coord = 5) and the lowest for confidence predictions when no object is present (λ noobj = 0.5).

## YOLO Loss Function — Part 4 (Loss Function is the classification loss)
![image](https://user-images.githubusercontent.com/117746681/224551686-f59c70b4-a15a-4a99-9141-58c1e52724e5.png)

It looks similar to a normal sum-squared error for classification, except for the 𝟙 obj term. This term is used because so we don’t penalize classification error when no object is present on the cell. (using the conditional class probablilty)

Moving to much awaited part...

# Implementation

## Dataset
The PASCAL Visual Object Classes (VOC) 2012 dataset contains 20 object categories including vehicles, household, animals, and other: aeroplane, bicycle, boat, bus, car, motorbike, train, bottle, chair, dining table, potted plant, sofa, TV/monitor, bird, cat, cow, dog, horse, sheep, and person. 
Each image in this dataset has pixel-level segmentation annotations, bounding box annotations, and object class annotations. This dataset has been widely used as a benchmark for object detection, semantic segmentation, and classification tasks. The PASCAL VOC dataset is split into three subsets: 1,464 images for training, 1,449 images for validation and a private testing set.

Download Dataset: (https://www.kaggle.com/datasets/734b7bcb7ef13a045cbdd007a3c19874c2586ed0b02b4afc86126e89d00af8d2)

## Pre-Requisites(prefered)
Software
1. VS CODE: >2017

Packages:
1. Python >=3.5
2. Torch > 1.10
3. Torchvision >0.1.0

## Files
1. <model.py>  => Consists of structure for CNN
2. <loss.py>  => YOLO Loss Function
3. <utils.py> => Intersection over Union, Non Max Supression, Mean average precision 
4. <train.py> => Train and load the model based on given hyperparameters
5. <test.py>  => Observe the results

## References
Parent Repo: (https://github.com/aladdinpersson/Machine-Learning-Collection)

YOLO Paper: (https://arxiv.org/abs/1506.02640)

Object Deetction: (https://acrocanthosaurus627.medium.com/object-detection-from-scratch-with-pytorch-yolov1-a56b49024c22)

Loss Function: (https://hackernoon.com/understanding-yolo-f5a74bbc7967)
