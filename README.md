# Multi-Label-Image-Classification

## Introduction
The goal of this challenge is to recognize objects from a number of visual object classes in realistic scenes (i.e. not pre-segmented objects). It is fundamentally a supervised learning learning problem in that a training set of labelled images is provided. The twenty object classes that have been selected are: 
 
Person: person 
Animal: bird, cat, cow, dog, horse, sheep 
Vehicle: aeroplane, bicycle, boat, bus, car, motorbike, train 
Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor 

For this task, we will use the dataset from Pascal VOC challenge 2007. Download the dataset from http://host.robots.ox.ac.uk/pascal/VOC/voc2007/ 
[Look at the Development Kit section for training data and Test Data section for test data]


## The task
Our task is to identify the classes present in a particular image. We will use three types of models for this task 

a.	Train twenty different neural networks. Each neural network detects a separate class, i.e, one neural network detects car, another one detects aeroplane, etc. 

b.	Using a shared neural network as a featuriser. Then connect it to task specific layers. For each task, there should be a single neuron in its task specific output layer. As discussed before, the architecture is similar to the following image: 
 
c.	Using a single neural network with twenty neurons in output layer. The architecture is also discussed while covering multi-task learning. 


## Points to be noted 
1.  The output for each image should be a vector of size 20. Each position representing a separate class, 0 denoting its absence and 1 denoting its presence. 
2.	Read the actual vector from the given annotation files. 
3.	Report average cosine similarity of actual vectors and output vectors. (For Test Data. Don’t use any of the test data for training) 
4.	Include some visual outputs in your report. (Some images and their actual annotations vs your output annotations; By annotations, I mean the classes that are present)  
5.	Include your remarks on which model is performing better and why. 
