2018.08.04

The code is slow and costs nearly 3 minutes for each (im,label), where im is a 1024 dimension complex vector. The training set in MNIST has 60000 such pairs and the expected running time is roughly 100 days thus unbearable. 

A model, obtained by training on the first 200 samples of MNIST (cost nearly 10 hours), performs badly on randomly choosing test sets. One obvious reason is that the model.bias is 5.334, where bias is supposed to be a part of a probabilty, and should be less than 1. The value 5.334 is obviously improper.

The bad performance may result from the small trainning set, or some mistakes in my code, or there are some space to improve the classifier in the paper. 

------------------------------------------------------------------------------
2018.08.01

A quantum classifier based on the paper 

"circuit-centric quantum classifier", Maria Schuld, Alex Bocharov, Krysta Svore and Nathan Wiebe. https://arxiv.org/abs/1804.00633

Aims to use mnist dataset to test the feasibiliy of the model.

Tips:
1.This code is based on classical simulation rather than using real quantum computers. I hope one day I can implement it on IBM's Quantum Experience ^o^ ~
2.For the limit of computation resource, in this code, function 'measure' using 'measure_fake', who gets the repeated measurement outcome by directly calculating the ratio of amplitutes, times the repeat_times. One can modify function 'measure' by using function 'measure_true', which simulates the random behaviour of quantum measurement but need expensive calculation( times repeated_times).    

Status: unfinished.
