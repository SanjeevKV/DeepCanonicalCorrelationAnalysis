# ===============================================================================
# (C) 2019 by Weiran Wang (weiranwang@ttic.edu), Qingming Tang (qmtang@ttic.edu),
# and Karen Livescu (klivescu@ttic.edu).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# ===============================================================================

import numpy as np
import os
import tensorflow as tf
import DCCA as dcca
from CCA import linCCA
from myreadinput import read_mnist


import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--Z", default=10, help="Dimensionality of features", type=int)
parser.add_argument("--dropprob", default=0.0, help="Dropout probability of networks.", type=float)
parser.add_argument("--checkpoint", default="./dcca_mnist", help="Path to saved models", type=str)
parser.add_argument("--batchsize", default=500, help="Number of samples in each minibatch", type=int)
parser.add_argument("--gpuid", default="0", help="ID of gpu device to be used", type=str)
args=parser.parse_args()

# Handle multiple gpu issues.
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid


if __name__ == "__main__":

    # Set random seeds.
    np.random.seed(0)
    tf.set_random_seed(0)
    
    # Obtain parsed arguments.
    Z=args.Z
    print("Dimensionality of shared variables: %d" % Z)
    dropprob=args.dropprob
    print("Dropout rate: %f" % dropprob)
    checkpoint=args.checkpoint
    print("Trained model will be saved at %s" % checkpoint)

    # Some other configurations parameters for mnist.
    learning_rate=0.001
    l2_penalty=0.0001
    rcov1=0.0001
    rcov2=0.0001
    
    # Define network architectures.
    network_architecture=dict(
        n_input1=784, # MNIST data input (img shape: 28*28)
        n_input2=784, # MNIST data input (img shape: 28*28)
        n_z=Z,  # Dimensionality of shared latent space
        F_hidden_widths=[1024, 1024, 1024, Z],
        F_hidden_activations=[tf.nn.tanh, tf.nn.tanh, tf.nn.tanh, None],
        G_hidden_widths=[1024, 1024, 1024, Z],
        G_hidden_activations=[tf.nn.tanh, tf.nn.tanh, tf.nn.tanh, None]
        )

    # First, build the model.
    model=dcca.DCCA(network_architecture, rcov1, rcov2, learning_rate, l2_penalty)
    saver=tf.train.Saver()
    
    # Second, load the saved moded, if provided.
    if checkpoint and os.path.isfile(checkpoint + ".meta"):
        print("loading model from %s " % checkpoint)
        saver.restore(model.sess, checkpoint)
        epoch=model.sess.run(model.epoch)
        print("picking up from epoch %d " % epoch)
        tunecost=model.sess.run(model.tunecost)
        print("tuning cost so far:")
        print(tunecost[0:epoch])
    else:
        print("checkpoint file not given or not existent!")

    # File for saving classification results.
    classfile=checkpoint + '_classify.mat'
    if os.path.isfile(classfile):
        print("Job is already finished!")
        exit(0)
    
    # Third, load the data.
    trainData,tuneData,testData=read_mnist()
    
    # Traning.
    model=dcca.train(model, trainData, tuneData, saver, checkpoint, batch_size=args.batchsize, max_epochs=50, save_interval=1, keepprob=(1.0-dropprob))

    # Satisfy constraint.
    FX1=model.compute_projection(1, trainData.images1)
    FX2=model.compute_projection(2, trainData.images2)
    A,B,m1,m2,_=linCCA(FX1, FX2, model.n_z, rcov1, rcov2)

    # SVM linear classification.
    print("Performing linear SVM!")
    trainData,tuneData,testData=read_mnist()
    best_error_tune=1.0
    from sklearn import svm
    for c in [0.1, 1.0, 10.0]:
        lin_clf=svm.SVC(C=c, kernel="linear")
        # train
        svm_x_sample=trainData.images1[::]
        svm_y_sample=np.reshape(trainData.labels[::], [svm_x_sample.shape[0]])
        svm_z_sample=np.matmul(model.compute_projection(1, svm_x_sample) - m1, A)
        lin_clf.fit(svm_z_sample, svm_y_sample)   
        # dev
        svm_x_sample=tuneData.images1
        svm_y_sample=np.reshape(tuneData.labels, [svm_x_sample.shape[0]])
        svm_z_sample=np.matmul(model.compute_projection(1, svm_x_sample) - m1, A)
        pred=lin_clf.predict(svm_z_sample)
        svm_error_tune=np.mean(pred != svm_y_sample)
        print("c=%f, tune error %f" % (c, svm_error_tune))
        if svm_error_tune < best_error_tune:
            best_error_tune=svm_error_tune
            bestsvm=lin_clf
    
    # test
    svm_x_sample=testData.images1
    svm_y_sample=np.reshape(testData.labels, [svm_x_sample.shape[0]])
    svm_z_sample=np.matmul(model.compute_projection(1, svm_x_sample) - m1, A)
    pred=bestsvm.predict(svm_z_sample)
    best_error_test=np.mean(pred != svm_y_sample)
    print("tuneerr=%f, testerr=%f" % (best_error_tune, best_error_test))

    
    # TSNE visualization and clustering.
    print("Visualizing shared variables!")
    trainData,tuneData,testData=read_mnist()
    z_tune=np.matmul(model.compute_projection(1, tuneData.images1) - m1, A)
    z_test=np.matmul(model.compute_projection(1, testData.images1) - m1, A)

    
    import scipy.io as sio
    sio.savemat(classfile, {'tuneerr':best_error_tune,  'testerr':best_error_test, 'z_tune':z_tune,  'z_test':z_test})


        
