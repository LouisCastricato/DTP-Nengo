# DifferenceTargetPropogationsNengo
Implemented DTP in nengo, thought I'd share my code.

Based off of http://arxiv.org/pdf/1412.7525v5.pdf


I couldn't get z eplison, or f epsilon to work properly since I don't know enough about nengo, but I did manage to get them working in theano (I may upload that code as well.)

Particularly, I implemented the autoencoder described in section 2.4. It doesn't calculate the inverse mapping yet, I'll be adding that sometime next week since I have it working but the code isn't clean enough to release. 

Not sure why it flips the mapping either

