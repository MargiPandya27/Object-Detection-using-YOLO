# Object Detection using YOLOv1 

**Before starting what is object detection...**

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

![Example](http://url/to/image.png)

**Confidence** reflects the presence or absence of an object of any class.
If object is present,
Confidence = Pr(Object) * IOU(pred, truth) 

If no object is not present,
confidence = 0

Prediction for bounding box in image will from each grid cell if makes B of those predictions, so there are in total S x S x B * 5 outputs related to bounding box predictions.
The output prediction will also consists of class probablities too  Pr(Class(i) | Object). This probability is conditioned on the grid cell containing one object. The network only predicts one set of class probabilities per cell, regardless of the number of boxes B. That makes S x S x C class probabilities in total

Adding the class predictions to the output vector, a **S x S x (B * 5 + C)** tensor is formed as output.

**Architecture**
![Architecture](http://url/to/image.png)

**Loss Function**

### YOLO Loss Function — Part 1 (Loss Function related to bounding predictions)

image

This equation computes the loss related to the predicted bounding box position (x,y). The function computes a sum over each bounding box predictor (j = 0.. B) of each grid cell (i = 0 .. S^2). 𝟙 obj is defined as follows:
1, If an object is present in grid cell i and the _j_th bounding box predictor is “responsible” for that prediction
0, otherwise

YOLO will predict multiple bounding boxes for a object. Only one predictor is assigned to be "responsible" it would be one with highest current IOU score.

    (x, y): predicted bounding box position,
    (x̂, ŷ): actual position from the training data.

### YOLO Loss Function — Part 2 (Loss Function related to dimensions)

image

This is the loss related to the predicted box width / height. The equation looks similar to the first one, except for the square root. What’s up with that? Quoting the paper again:

Our error metric should reflect that small deviations in large boxes matter less than in small boxes. To partially address this, the square root of the bounding box width and height are predicted instead of the width and height directly.

### YOLO Loss Function — Part 3 (Loss Function related to confidence score)

image

Here, the loss associated with the confidence score for each bounding box predictor is computed. 

    C: confidence score 
    Ĉ: is the intersection over union of the predicted bounding box with the ground truth.
    𝟙 obj: 1 when there is an object in the cell, 
           0 otherwise. 
    𝟙 noobj is the opposite.

λ parameters that are used to differently weight parts of the loss functions. This is necessary to increase model stability. The highest penalty is for coordinate predictions (λ coord = 5) and the lowest for confidence predictions when no object is present (λ noobj = 0.5).

## YOLO Loss Function — Part 4 (Loss Function is the classification loss)

image

It looks similar to a normal sum-squared error for classification, except for the 𝟙 obj term. This term is used because so we don’t penalize classification error when no object is present on the cell. (using the conditional class probablilty)













