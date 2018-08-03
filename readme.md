A quantum classifier based on the paper 

"circuit-centric quantum classifier", Maria Schuld, Alex Bocharov, Krysta Svore and Nathan Wiebe. https://arxiv.org/abs/1804.00633

Aims to use mnist dataset to test the feasibiliy of the model.

Tips:

1.This code is based on classical simulation rather than using real quantum computers. I hope one day I can implement it on IBM's Quantum Experience ^o^ ~

2.For the limit of computation resource, in this code, function 'measure' using 'measure_fake', who gets the repeated measurement outcome by directly calculating the ratio of amplitutes, times the repeat_times. One can modify function 'measure' by using function 'measure_real', which simulates the random behaviour of quantum measurement but need expensive calculation( times repeated_times).    

Status: unfinished.
