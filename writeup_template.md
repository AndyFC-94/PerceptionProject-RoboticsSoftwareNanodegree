# Project: Perception Pick & Place

---

![demo-1](https://user-images.githubusercontent.com/20687560/28748231-46b5b912-7467-11e7-8778-3095172b7b19.png)


[//]: # (Image References)

[exercise1]: ./images/Exercise-1.png
[exercise2]: ./images/Exercise-2.png
[exercise3]: ./images/Exercise-3.png
[world1-training]: ./images/world1-training.png
[world1-recognition]: ./images/world1-recognition.png
[world2-training]: ./images/world2-training.png
[world2-recognition]: ./images/world2-recognition.png
[world3-training]: ./images/world3-training.png
[world3-recognition]: ./images/world3-recognition.png

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.
* I implemented a pipeline to separate the objects from the table. 
* [RANSAC.py](./Exercise-1/RANSAC.py)

* The following algorithms were implemented: 
* * Voxel Grif Filter
* * PassThrough filter
* * RANSAC plane segmentation

Extracted objects

![alt text][exercise1]

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  
* Here, I implemented clustering for segmentation of the objects.
* [Segmentation.py](./Exercise-2/sensor_stick/scripts/segmentation.py)
* I implemented Euclidean Clustering.
* I set the tolerance, minimum cluster size and maximum cluster size.

```python
    ec.set_ClusterTolerance(0.05)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(1500)
```
* Finally, I asigned a color to each segment

Colored objects

![alt text][exercise2]

#### 2. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.

* For object recognition I implemented Support Vector Machine. 
* [Object_recognition.py](./Exercise-3/sensor_stick/scripts/object_recognition.py)

![alt text][exercise3]

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

* I implemented all the techniques from previous exercises. 
* The main goal is to train three different worlds and generate output.yaml files.
* First, I trained each world with capture_features.py and train_svm.py.
* In capture_features.py I set the number of attempts to 40 and set hsv to True.
* In train_svm.py  I work with a 'linear' kernel.
* I have the following results:

WORLD 1
![alt text][world1-training]

WORLD 2
![alt text][world2-training]

WORLD 3
![alt text][world3-training]

* I implemented my solution in pr2_project.py file. 

WORLD 1
![alt text][world3-recognition]

WORLD 2
![alt text][world3-recognition]

WORLD 3
![alt text][world3-recognition]

* Finally, I generate the following output.yaml files: 

* [OUTPUT-1.yml](./RoboND-Perception-Project/pr2_robot/scripts/output_1.yaml)
* [OUTPUT-2.yml](./RoboND-Perception-Project/pr2_robot/scripts/output_2.yaml)
* [OUTPUT-3.yml](./RoboND-Perception-Project/pr2_robot/scripts/output_3.yaml)


### Results

* I  Identified 3/3 objects in world1, 5/5 objects in world 2, and 7/8 objects in world 3.
* I think I have to modify the Euclidean Clustering paremeters to recognize all the objects in world3.
* I would like to perform the challenge, but my PC requires too much resources to run the simulator.


