# MNIST-IN-C
MNIST Handwritten Digit Classifier in a Single C File.

## Introduction

The purpose of this project was to understand how a neural network
can solve problems. The implementation was chosen to avoid code
copying, and to have a greater understanding of the underlying math
the computer must perform to be able to classify digits.

## Method

The neural network developed has two layers and 
implements SGD - Stochastic Gradient Descent. 
No libraries were used other than stdio.h

The neural net is heavily based on the first chapters of Michael Nielsen's book - 
http://neuralnetworksanddeeplearning.com/. And by correlation the 
great series of videos by 3b1b - https://youtu.be/aircAruvnKk.

The original data is available at http://yann.lecun.com/exdb/mnist/. 

## Results

Using a batch size of around 10 and two layers with 30 nodes on the second layer
I was able to achieve around 95% correct results over the test cases in the 10th
epoch.

## Building in Linux

To build in Linux all you need to run is build.sh in the code folder.
Provided you have gcc installed. 

## Building in Windows

Assuming you have cl enabled - such as through vcvarsall.bat - all you need to do
is run build.bat on the code folder.

## Running in Linux

To run the code in Linux you just need to run main.out from the build folder. Sample output: 

![sample_output_linux](https://user-images.githubusercontent.com/47281204/156784612-f9389b3b-efc3-41aa-9613-9bb4b9a26436.png)

## Running in Windows

To run the code in Windows you just need to run main.exe from the build folder. Sample output:

![sample_output_win](https://user-images.githubusercontent.com/47281204/156784778-34f0dbc9-3235-4c17-93d8-552b454bcb3c.png)
