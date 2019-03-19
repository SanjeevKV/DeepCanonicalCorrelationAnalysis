import numpy as np
import math
import tensorflow as tf
from CCA import linCCA, CCA_loss

########################## CONSTRUCT AN DENSE DNN ##############################
def dnn(inputs, D_in, layer_widths, layer_activations, keepprob, name, variable_reuse, initializer):
    """ Expects flattened inputs.
    """

    width=D_in
    with tf.variable_scope(name, reuse=variable_reuse, initializer=initializer):
        activation=inputs
        for i in range(len(layer_widths)):
            # print("\tLayer %d ..." % (i+1))
            activation=tf.nn.dropout(activation, keepprob)
            weights=tf.get_variable("weights_layer_" + str(i+1), [width, layer_widths[i]])
            biases=tf.get_variable("biases_layer_" + str(i+1), [layer_widths[i]])
            activation=tf.add(tf.matmul(activation, weights), biases)
            if layer_activations[i] is not None:
                activation=layer_activations[i](activation)
            width=layer_widths[i]
    return activation


class DCCA(object):
    
    def __init__(self, architecture, rcov1=0, rcov2=0, learning_rate=0.0001, l2_penalty=0.0):

        # Save the architecture and parameters.
        self.network_architecture=architecture
        self.l2_penalty=l2_penalty
        self.learning_rate=tf.Variable(learning_rate,trainable=False)
        # self.learning_rate=learning_rate
        self.rcov1=rcov1
        self.rcov2=rcov2
        self.n_input1=n_input1=architecture["n_input1"]
        self.n_input2=n_input2=architecture["n_input2"]
        self.n_z=n_z=architecture["n_z"]
        
        # Tensorflow graph inputs.
        self.batchsize=tf.placeholder(tf.float32)
        self.x1=tf.placeholder(tf.float32, [None, n_input1])
        self.x2=tf.placeholder(tf.float32, [None, n_input2])
        self.keepprob=tf.placeholder(tf.float32)

        # Variables to record training progress.
        self.epoch=tf.Variable(0, trainable=False)
        self.tunecost=tf.Variable(tf.zeros([1000]), trainable=False)
        
        # Initialize network weights and biases.
        initializer=tf.random_uniform_initializer(-0.05, 0.05)
        
        # Use the recognition network to obtain the Gaussian distribution (mean and log-variance) of latent codes.
        print("Building view 1 projection network F ...")
        self.FX1=dnn(self.x1, self.n_input1, architecture["F_hidden_widths"], architecture["F_hidden_activations"], self.keepprob, "F", None, initializer)

        print("Building view 2 projection network G ...")
        self.FX2=dnn(self.x2, self.n_input2, architecture["G_hidden_widths"], architecture["G_hidden_activations"], self.keepprob, "G", None, initializer)

        print("Covariance regularizations: [%f, %f]" % (rcov1, rcov2))
        self.canoncorr=CCA_loss(self.FX1, self.FX2, self.batchsize, n_z, n_z, n_z, self.rcov1, self.rcov2)
        
        # Weight decay.
        self.weightdecay=tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

        # Define cost and use the ADAM optimizer.
        self.cost= - self.canoncorr + l2_penalty * self.weightdecay
        # self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        self.optimizer=tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(self.cost)
        
        
        # Initializing the tensor flow variables and launch the session.
        init=tf.global_variables_initializer()
        self.sess=tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(init)


    def assign_lr(self, lr):
        self.sess.run(tf.assign(self.learning_rate, lr))

    def assign_epoch(self, EPOCH_VALUE):
        self.sess.run(tf.assign(self.epoch, EPOCH_VALUE))
    
    def assign_tunecost(self, TUNECOST_VALUE):
        self.sess.run(tf.assign(self.tunecost, TUNECOST_VALUE))

    
    def partial_fit(self, X1, X2, keepprob):
        
        # Train model based on mini-batch of input data. Return cost of mini-batch.
        opt, cost=self.sess.run( [self.optimizer, self.cost], feed_dict={self.x1: X1, self.x2: X2, self.batchsize: X1.shape[0], self.keepprob: keepprob})
        return cost
    
    
    def compute_projection(self, view, X):
        
        N=X.shape[0]
        Dout=self.n_z
        
        FX=np.zeros([N, Dout], dtype=np.float32)
        batchsize=5000
        for batchidx in range(np.ceil(N / batchsize).astype(int)):
            idx=range( batchidx*batchsize, min(N, (batchidx+1)*batchsize) )
            if view==1:
                tmp=self.sess.run(self.FX1, feed_dict={self.x1: X[idx,:], self.keepprob: 1.0})
            else:
                tmp=self.sess.run(self.FX2, feed_dict={self.x2: X[idx,:], self.keepprob: 1.0})
            FX[idx,:]=tmp
        return FX

    
def train(model, trainData, tuneData, saver, checkpoint, batch_size=100, max_epochs=10, save_interval=5, keepprob=1.0):
    
    epoch=model.sess.run(model.epoch)
    TUNECOST=model.sess.run(model.tunecost)
    lr=model.sess.run(model.learning_rate)
    
    n_samples=trainData.num_examples
    total_batch=int(math.ceil(1.0 * n_samples / batch_size))
    
    # Training cycle.
    while epoch < max_epochs:
        print("Current learning rate %f" % lr)
        avg_cost=0.0
        
        # Loop over all batches.
        NANERROR=False
        for i in range(total_batch):
            batch_x1, batch_x2, _=trainData.next_batch(batch_size)
            
            # Fit training using batch data.
            cost=model.partial_fit(batch_x1, batch_x2, keepprob)
            print("minibatch %d/%d: cost=%f" % (i+1, total_batch, cost))
            
            # Compute average loss.
            if not np.isnan(cost):
                avg_cost +=cost / n_samples * batch_size
            else:
                NANERROR=True
                break

        if NANERROR:
            print("Loss is nan. Reverting to previously saved model ...")
            saver.restore(model.sess, checkpoint)
            epoch = model.sess.run(model.epoch)
            TUNECOST = model.sess.run(model.tunecost)
            continue

        # Compute validation error, turn off dropout.
        FXV1=model.compute_projection(1, tuneData.images1)
        FXV2=model.compute_projection(2, tuneData.images2)
        _, _, _, _, tune_E=linCCA(FXV1, FXV2, model.n_z, model.rcov1, model.rcov2)
        tune_canoncorr=np.sum(tune_E)
        TUNECOST[epoch]=-tune_canoncorr
        
        # Display logs per epoch step.
        epoch=epoch+1
        print("Epoch: %04d, train regret=%12.8f, tune cost=%12.8f" % (epoch, avg_cost, -tune_canoncorr))
        
        if (checkpoint) and (epoch % save_interval == 0):
            model.assign_epoch(epoch)
            model.assign_tunecost(TUNECOST)
            save_path=saver.save(model.sess, checkpoint)
            print("Model saved in file: %s" % save_path)
            
    return model


"""


############################# Visualize shared information.
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

import scipy.io as sio
M=sio.loadmat('../MNIST.mat')
y_sample=np.squeeze(M["testLabels"][::2])

F=sio.loadmat("dcca_classify.mat")
z_test=F["z_test"]

from sklearn.manifold import TSNE
tsne=TSNE(perplexity=20, n_components=2, init="pca", n_iter=3000)
z_tsne=tsne.fit_transform( np.asfarray(z_test[::2], dtype="float") )

COLORS=[
[1.0000, 0,      0     ],
[0,      1.0000, 0     ], 
[0,      0,      1.0000],
[1.0000, 0,      1.0000],
[0.9569, 0.6431, 0.3765],
[0.4000, 0.8039, 0.6667],
[0.5529, 0.7137, 0.8039],
[0.8039, 0.5882, 0.8039],
[0.7412, 0.7176, 0.4196],
[0,      0,      0     ]]

plt.figure(11,figsize=(10,10))
hhs=[]
for i in range(10):
    idx=np.argwhere(y_sample==(i+1))
    h=plt.plot(z_tsne[idx, 0], z_tsne[idx, 1], "o", c=COLORS[i], markersize=8.0)
    hhs.append(h[0])

plt.legend(hhs, ["1","2","3","4","5","6","7","8","9", "0"] )
plt.tight_layout()
plt.axis('off')
plt.title("DCCA")


############################# Visualized private information.
h1, _= model.transform_private(1, x1_sample)

from sklearn.manifold import TSNE
tsne=TSNE(perplexity=20, n_components=2, init="pca", n_iter=3000)
h1_tsne=tsne.fit_transform( np.asfarray(h1, dtype="float") )

plt.figure(12,figsize=(10,10))
hhs=[]
for i in range(10):
    idx=np.argwhere(y_sample==(i+1))
    h=plt.plot(h1_tsne[idx, 0], h1_tsne[idx, 1], "o", c=COLORS[i], markersize=8.0)
    hhs.append(h[0])

plt.legend(hhs, ["1","2","3","4","5","6","7","8","9", "0"] )
plt.tight_layout()
plt.axis('off')
# plt.savefig("MNIST_VCCA_2D.eps")
plt.title("private")



############################# Visualized private information.
h2, _= model.transform_private(2, x2_sample)

from sklearn.manifold import TSNE
tsne=TSNE(perplexity=20, n_components=2, init="pca", n_iter=3000)
h2_tsne=tsne.fit_transform( np.asfarray(h2, dtype="float") )

plt.figure(13,figsize=(10,10))
hhs=[]
for i in range(10):
    idx=np.argwhere(y_sample==(i+1))
    h=plt.plot(h2_tsne[idx, 0], h2_tsne[idx, 1], "o", c=COLORS[i], markersize=8.0)
    hhs.append(h[0])

plt.legend(hhs, ["1","2","3","4","5","6","7","8","9", "0"] )
plt.tight_layout()
plt.axis('off')
# plt.savefig("MNIST_VCCA_2D.eps")
plt.title("view 2 private")

import scipy.io as sio
sio.savemat('vcca_shared_emb_30.mat', {'z':z, 'z_tsne':z_tsne, 'h1':h1, 'h1_tsne':h1_tsne, 'h2':h2, 'h2_tsne':h2_tsne})



############################# VISUALIZE INPUTS.
tsne=TSNE(perplexity=20, n_components=2, init="pca", n_iter=5000)
xinput=tsne.fit_transform_shared( np.asfarray(x1_sample, dtype="float") )

plt.figure(22,figsize=(10,10))
hhs=[]
for i in range(10):
    idx=np.argwhere(y_sample==(i+1))
    h=plt.plot(xinput[idx, 0], xinput[idx, 1], "o", c=COLORS[i])
    hhs.append(h[0])

plt.legend(hhs, ["1","2","3","4","5","6","7","8","0"] )
plt.tight_layout()
plt.savefig("INPUT1_tsne.eps")



############################# LINEAR SVM CLASSIFICATION.
trainData,tuneData,testData=read_mnist()
from sklearn import svm
lin_clf=svm.SVC(C=10, kernel="linear")

# train
svm_x_sample=trainData.images1[::]
svm_y_sample=np.reshape(trainData.labels[::], [svm_x_sample.shape[0]])
svm_z_sample, svm_z_std=model.transform_shared(1, svm_x_sample)
svm_z_sample_private, _=model.transform_private(1, svm_x_sample)
# svm_z_sample=np.concatenate([svm_z_sample, svm_z_sample_private], 1)
lin_clf.fit(svm_z_sample, svm_y_sample)

# predict
svm_x_sample=tuneData.images1
svm_y_sample=np.reshape(tuneData.labels, [svm_x_sample.shape[0]])
svm_z_sample, svm_z_std=model.transform_shared(1, svm_x_sample)
svm_z_sample_private, _=model.transform_private(1, svm_x_sample)
# svm_z_sample=np.concatenate([svm_z_sample, svm_z_sample_private], 1)
pred=lin_clf.predict(svm_z_sample)
np.mean(pred != svm_y_sample)

svm_x_sample=testData.images1
svm_y_sample=np.reshape(testData.labels, [svm_x_sample.shape[0]])
svm_z_sample, svm_z_std=model.transform_shared(1, svm_x_sample)
svm_z_sample_private, _=model.transform_private(1, svm_x_sample)
# svm_z_sample=np.concatenate([svm_z_sample, svm_z_sample_private], 1)
pred=lin_clf.predict(svm_z_sample)
np.mean(pred != svm_y_sample)

from sklearn.metrics import confusion_matrix
confusion_matrix(svm_y_sample, pred)
"""

"""
############################# Visualize train.
trainData,tuneData,testData=read_mnist()
x1_sample=trainData.images1[0::250]
x2_sample=trainData.images2[0::250]
x1_reconstruct, x2_reconstruct=model.reconstruct(1, x1_sample, x2_sample)

for i in range(0,200,20):
    # Plot one figure.
    plt.figure(1, figsize=(10.0,10.0))
    plt.imshow(np.transpose(x1_sample[i].reshape(28, 28)), vmin=0.0, vmax=1.0)
    # plt.title("view 1 input")
    plt.set_cmap("gray")
    plt.tight_layout()
    plt.axis("off")
    figname="shared_train_"+str(i)+"_view1_input.eps"
    plt.savefig(figname)
    raw_input()
    plt.close(1)

for i in range(0,200,20):
    plt.figure(2, figsize=(10.0,10.0))
    plt.imshow(np.transpose(x1_reconstruct["mean"][i].reshape(28, 28)), vmin=0, vmax=1)
    # plt.title("view 1 recons mean")
    plt.set_cmap("gray")
    plt.tight_layout()
    plt.axis("off")
    figname="shared_train_"+str(i)+"_view1_recon_mean.eps"
    plt.savefig(figname)
    raw_input()
    plt.close(2)

for i in range(0,200,20):
    plt.figure(3, figsize=(10.0,10.0))
    plt.imshow(np.transpose(x2_sample[i].reshape(28, 28)))
    # plt.title("view 2 input")
    plt.set_cmap("gray")
    plt.tight_layout()
    plt.axis("off")
    figname="shared_train_"+str(i)+"_view2_input.eps"
    plt.savefig(figname)
    raw_input()
    plt.close(3)

for i in range(0,200,20):
    plt.figure(4, figsize=(10.0,10.0))
    plt.imshow(np.transpose(x2_reconstruct["mean"][i].reshape(28, 28)))
    # plt.title("view 2 recons mean")
    plt.set_cmap("gray")
    plt.tight_layout()
    plt.axis("off")
    figname="shared_train_"+str(i)+"_view2_recon_mean.eps"
    plt.savefig(figname)
    raw_input()
    plt.close(4)

for i in range(0,200,20):
    plt.figure(5, figsize=(10.0,10.0))
    plt.imshow(np.transpose(x2_reconstruct["std"][i].reshape(28, 28)))
    # plt.title("view 2 recons std")
    plt.set_cmap("gray")
    plt.tight_layout()
    plt.axis("off")
    figname="shared_train_"+str(i)+"_view2_recon_std.eps"
    plt.savefig(figname)
    raw_input()
    plt.close(5)


"""
