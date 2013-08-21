NNKit
=====

NNKit is the very beginnings of an object oriented neural network exploration kit.  So far, it
implements a few flavors of backprop and has some basic reporting.  It is slow, for now, as no
attempt has yet been made to optimize transfers to and from the GPU.

It is written in Python and built on primarily Theano, Numpy, and CUDA; it also uses matplotlib
and OpenCV for some non-core tasks: charts, sample data set generation, etc.

To run the examples, clone this repo, cd to nnkit/src and type 'make'.  Then cd to
nnkit/build/nnkit/examples and type 'python ex01_classify.py' to run the input classification
example, or 'python ex02_bestfit.py' to run the best fit example.

The output of the latter can be seen in the directory nnkit/build/nnkit/examples/ex02_images; in
particular compare ex02_images/all/output_comparison.0.png (the initial output) with
ex02_images/all/output_comparison.99.png (the last).  The first column is the training input image,
the second is the label, and the third is the network output.

There are lots of comments throughout the code, and I've tried to code this in a clear,
clever-free style.