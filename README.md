# Multi-Label-Image-Classification

## Introduction
The goal of this challenge is to recognize objects from a number of visual object classes in realistic scenes (i.e. not pre-segmented objects). It is fundamentally a supervised learning learning problem in that a training set of labelled images is provided. The twenty object classes that have been selected are: 
 
Person: person 
Animal: bird, cat, cow, dog, horse, sheep 
Vehicle: aeroplane, bicycle, boat, bus, car, motorbike, train 
Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor 

For this task, we will use the dataset from Pascal VOC challenge 2007. Download the dataset from http://host.robots.ox.ac.uk/pascal/VOC/voc2007/ <br>Look at the Development Kit section for training data and Test Data section for test data.


## The task
Our task is to identify the classes present in a particular image. Like the following ones (Final Outputs)- 

![sample1](https://github.com/TasnimAhmedEee/Multi-Label-Image-Classification/blob/master/sample1.png)

![sample2](https://github.com/TasnimAhmedEee/Multi-Label-Image-Classification/blob/master/sample2.png)





We will use three types of models for this task 

a.	Training twenty(20) different neural networks. Each neural network detects a separate class, i.e, one neural network detects car, another one detects aeroplane, etc. 

b.	Using a shared neural network as a featuriser. Then connect it to task specific layers. For each task, there should be a single neuron in its task specific output layer. As discussed before, the architecture is similar to the following image:

![multi-task-nn](https://github.com/TasnimAhmedEee/Multi-Label-Image-Classification/blob/master/multi-task-nn.png)
 
c.	Using a single neural network with twenty neurons in output layer. Each neuron of output-layer represents a class here.

![shared-nn](https://github.com/TasnimAhmedEee/Multi-Label-Image-Classification/blob/master/shared-nnn.png)


## Points to be noted 
1.  The output for each image should be a vector of size 20. Each position representing a separate class, 0 denoting its absence and 1 denoting its presence. 
2.	Reading the actual vector from the given annotation files. 
3.	Reporting average cosine similarity of actual vectors and output vectors. (For Test Data. Don’t use any of the test data for training) 
4.	Including visual outputs in your report. (Some images and their actual annotations vs your output annotations; By annotations, I mean the classes that are present)  
5.	Including remarks on which model is performing better and why. 

Python Dependencies:

    NumPy
    Pandas
    Sklearn
    Tensorflow
    Matplotlib
    Keras

Was inspired from-<br>
https://blog.manash.me/multi-task-learning-in-keras-implementation-of-multi-task-classification-loss-f1d42da5c3f6

--Thanks a lot for viewing<br><br></p>

N.B.: Jupyter-notebook files are large and may not be displayed properly in GitHub Use the online Notebook viewer: https://nbviewer.jupyter.org/ to view the notebooks.

