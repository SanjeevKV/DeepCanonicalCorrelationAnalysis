import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
#tf.compat.v1.enable_eager_execution()

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

import pandas as pd
import numpy as np
import sys
import keras
import pickle

from keras.models import Sequential, Model
from keras.layers import SeparableConv1D, AveragePooling1D, CuDNNLSTM, Concatenate, Lambda, Reshape, Input, Bidirectional, TimeDistributed, GlobalAveragePooling1D, SpatialDropout1D, Conv1D, BatchNormalization, MaxPooling1D, Dense, Activation, Flatten, LSTM, Dropout, Layer
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
import keras.backend as K

BatchSize = 128
eps_eig = 1e-6

"""
files - List of files to use for generating Ids
lp - Starting point of the data frames
up - End point of the data frames
duration - Number of frames to constitute one data point
l_chuck - Number of frames to chuck at the beginning
u_chuck - Number of frames to chuck at the end
"""
def IdAssigner(files, lp, up, duration = 120, l_chuck = 0, u_chuck = 0, shift = 0):
    counter = 1
    id_dictionary = {}
    for fi in files:
        df = pd.read_csv(fi)
        num_frames = len(df)
        clb = l_chuck + int(lp * num_frames) #Current lower bound
        cub = clb + duration
        print(fi)
        
        while num_frames - int( (1 - up) * num_frames ) - u_chuck >= cub:
            id_dictionary[counter] = df.iloc[clb:cub, :]
            counter += 1
            clb += shift
            cub += shift
            
    return id_dictionary


"""
files - List of files to use for generating Ids
lp - Starting point of the data frames
up - End point of the data frames
duration - Number of frames to constitute one data point
l_chuck - Number of frames to chuck at the beginning
u_chuck - Number of frames to chuck at the end
"""
def IdAssignerMotion(files, lp, up, columns, duration = 120, l_chuck = 0, u_chuck = 0, shift = 0):
    counter = 1
    id_dictionary = {}
    for fi in files:
        df = pd.read_csv(fi)
        df = df[columns]
        num_frames = len(df)
        clb = l_chuck + int(lp * num_frames) #Current lower bound
        cub = clb + duration
        print(fi)
        
        while num_frames - int( (1 - up) * num_frames ) - u_chuck >= cub:
            id_dictionary[counter] = df.iloc[clb:cub, :]
            counter += 1
            clb += shift
            cub += shift
            
    return id_dictionary

def LinesToList(location):
    with open(location) as f:
        all_lines = f.read()
    lines = all_lines.split("\n")
    return lines[:-1]
    

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, ids_dictionary, labels_dictionary, batch_size=BatchSize, dim=(120,), n_channels_speech=13, n_channels_motion = 3,
                 n_classes=3, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels_dictionary = labels_dictionary
        self.list_IDs = list(labels_dictionary.keys())
        self.n_channels_speech = n_channels_speech
        self.n_channels_motion = n_channels_motion
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.ids_dictionary = ids_dictionary

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = indexes.copy()#[self.list_IDs[k] for k in indexes]

        # Generate data
        X1,X2, y = self.__data_generation(list_IDs_temp)
        #print(type(y))
        return [X1, X2], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = self.list_IDs #In turn, list_IDs also get shuffled
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X1 = np.empty((self.batch_size, *self.dim, self.n_channels_speech))
        X2 = np.empty((self.batch_size, *self.dim, self.n_channels_motion))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X1[i,] = self.__GetOneDataPoint(ID)#np.load('data/' + ID + '.npy')

            # Store sample
            X2[i,] = self.__GetOneMotionPoint(ID)#np.load('data/' + ID + '.npy')
            # Store class
            y[i] = 1#self.__GetLabel(ID)
        #print(type(y))
        #print(X1.shape, X2.shape, y.shape)
        return X1, X2, y #keras.utils.to_categorical(y, num_classes=self.n_classes)
        
    def __GetOneDataPoint(self, ID):
        df = self.ids_dictionary[ID]
        return np.array(df)

    def __GetOneMotionPoint(self, ID):
        df = self.labels_dictionary[ID]
        return np.array(df)
        
    def __GetLabel(self, ID):  
        df = self.labels_dictionary[ID]                          
        return self.__VelocityGrader(df)


def CanonCorr(H1, H2, N, d1, d2, dim, rcov1 = 0.0000, rcov2 = 0.000):

    # Remove mean.
    m1 = tf.reduce_mean(H1, axis=0, keep_dims=True)
    H1 = tf.subtract(H1, m1)

    m2 = tf.reduce_mean(H2, axis=0, keep_dims=True)
    H2 = tf.subtract(H2, m2)
    #print("m2:", m2.shape)
    d1, d2, dim = int(d1), int(d2), int(dim)
    #print(d1,d2,dim)
    #print(type(H1), type(N), type(rcov1), type(tf.eye(d1)))
    print(H1.shape, H2.shape)
    S11 = tf.matmul(tf.transpose(H1), H1) / (N-1) + rcov1 * tf.eye(d1)
    S22 = tf.matmul(tf.transpose(H2), H2) / (N-1) + rcov2 * tf.eye(d2)
    S12 = tf.matmul(tf.transpose(H1), H2) / (N-1)
#    print("PRINTING TENSORFLOW")
#    print(tf.reduce_sum(S11).numpy(), tf.reduce_sum(S22).numpy(), tf.reduce_sum(S12).numpy())
    E1, V1 = tf.self_adjoint_eig(S11)
    E2, V2 = tf.self_adjoint_eig(S22)

    #print(tf.reduce_sum(E1).numpy(), tf.reduce_sum(V1).numpy(), tf.reduce_sum(E2).numpy(), tf.reduce_sum(V2).numpy())
    # For numerical stability.
    idx1 = tf.where(E1>eps_eig)[:,0]
    E1 = tf.gather(E1, idx1)
    V1 = tf.gather(V1, idx1, axis=1)

    idx2 = tf.where(E2>eps_eig)[:,0]
    E2 = tf.gather(E2, idx2)
    V2 = tf.gather(V2, idx2, axis=1)

    K11 = tf.matmul(tf.matmul(V1, tf.diag(tf.reciprocal(tf.sqrt(E1)))), tf.transpose(V1))
    K22 = tf.matmul(tf.matmul(V2, tf.diag(tf.reciprocal(tf.sqrt(E2)))), tf.transpose(V2))
    T = tf.matmul(tf.matmul(K11, S12), K22)
    #print("T: ", T.shape)

    # Eigenvalues are sorted in increasing order.
    E3, U = tf.self_adjoint_eig(tf.matmul(T, tf.transpose(T)))
    #print("E3: ", E3.shape)
    idx3 = tf.where(E3 > eps_eig)[:, 0]
    #print("idx3: ", idx3.shape)
    # This is the thresholded rank.
    dim_svd = tf.cond(tf.size(idx3) < dim, lambda: tf.size(idx3), lambda: dim)
    #dim_svd = 0
    #print("dim_svd: ", dim_svd.shape)
    #print("reduce_sum", tf.reduce_sum(tf.sqrt(E3[-dim_svd:]), keepdims = True).shape)
    #print(tf.reduce_sum(K11).numpy(), tf.reduce_sum(K22).numpy(), tf.reduce_sum(T).numpy(), tf.reduce_mean(tf.sqrt(E3)).numpy(), tf.reduce_mean(tf.sqrt(E3[-dim_svd:])).numpy())
    return tf.reduce_sum(tf.sqrt(E3[-dim_svd:]), keepdims = True), E3, dim_svd

"""
Keras custom Lambda layer function for combining output of two branches of a graph
"""
def CombineSpeechMotion(speech_motion):
    speech, motion = speech_motion[0], speech_motion[1]
    print(type(speech), type(motion))
    speech = tf.reshape(speech, [tf.shape(speech)[0] * tf.shape(speech)[1], speech.shape[-1] ])
    motion = tf.reshape(motion, [tf.shape(motion)[0] * tf.shape(motion)[1], motion.shape[-1] ])
    #print(speech.shape, motion.shape)
    BatchSize = tf.cast(tf.shape(speech)[0], dtype = tf.float32)
    corr,_,_ = CanonCorr(speech, motion, BatchSize, speech.shape[-1], motion.shape[-1], speech.shape[-1], 0.0001, 0.0001)
    #print(corr.shape, type(corr))
    return corr 

def NegativeCanonCorr(y_true, y_pred):
    #print(y_true.shape, y_pred.shape)
    print(- K.mean(y_pred))
    return - K.mean(y_pred)

def CommonNetwork(data_in):
    data_out = BatchNormalization()(data_in)
    #data_out = data_in
    data_out = Flatten()(data_out)
    #data_out = Dense(64, activation = 'linear')(data_out)
    #return data_out
    #data_out = BatchNormalization()(data_out)
    data_out = Dense(32, activation = 'relu')(data_out)
    data_out = BatchNormalization()(data_out)
    #data_out = Dense(24, activation = 'relu')(data_out)
    #data_out = BatchNormalization()(data_out)
    data_out = Dense(16, activation = 'relu')(data_out)
    data_out = BatchNormalization()(data_out)
    data_out = Dense(4, activation = 'relu')(data_out)
    data_out = BatchNormalization()(data_out)
    return data_out            

def SpeechNetwork(speech_input):
    #Network for processing speech data
    speech_in = BatchNormalization()(speech_input)
    speech = Conv1D(24, 17, activation = 'relu', name = "speech_convolution_1")(speech_in)
    speech = BatchNormalization()(speech)
    speech = MaxPooling1D(pool_size = 3)(speech)
    speech = Conv1D(24, 17, activation = 'relu', name = "speech_convolution_2")(speech)
    speech = BatchNormalization()(speech)
    speech = MaxPooling1D(pool_size = 3)(speech)
    speech = Conv1D(24, 17, activation = 'relu', name = "speech_convolution_3")(speech)
    speech = BatchNormalization()(speech)
    speech = MaxPooling1D(pool_size = 3)(speech)
    #speech = Reshape((-1,speech.shape[-1]))(speech)
    #print("Before LSTM", speech.shape)
    #speech = Bidirectional(LSTM(units = 24, return_sequences = False, name = "speech_lstm"))(speech)
    #speech = BatchNormalization()(speech)
    #speech = GlobalAveragePooling1D(name = "speech_global_averager")(speech)
    #speech = BatchNormalization()(speech)
    return speech

def MotionNetwork(motion_input):
    #rf = (241 - k - p * (k-1) - p**2 * ( k-1)) / p**3 : 240 corresponds to a receptive field of 2 seconds
    #p and k could be modified such that rf=1 for it to have a receptive field of 2 seconds.
    #(p=2,k=34) and (p=3, k=17) are some of the apirs possible
    #Network for processing head motion data
    motion_in = BatchNormalization()(motion_input)
    motion = Conv1D(24, 17, activation = 'relu', name = "motion_convolution_1")(motion_in)
    motion = BatchNormalization()(motion)
    motion = MaxPooling1D(pool_size = 3)(motion)
    motion = Conv1D(24, 17, activation = 'relu', name = "motion_convolution_2")(motion)
    motion = BatchNormalization()(motion)
    motion = MaxPooling1D(pool_size = 3)(motion)
    motion = Conv1D(24, 17, activation = 'relu', name = "motion_convolution_3")(motion)
    motion = BatchNormalization()(motion)
    motion = MaxPooling1D(pool_size = 3)(motion)
    #motion = Reshape((-1, motion.shape[-1]))(motion)
    #motion = Bidirectional(LSTM(units = 24, return_sequences = False, name = "motion_lstm"))(motion)
    #motion = BatchNormalization()(motion)
    #motion = GlobalAveragePooling1D(name = "motion_global_averager")(motion)
    #motion = BatchNormalization()(motion)
    return motion

def functional_CNN_pool_lstm_glob( input_shape_speech, input_shape_motion, num_classes ):
    print("Speech: ", input_shape_speech)
    print("Motion: ", input_shape_motion)
    speech_input = Input(shape = input_shape_speech, name = "speech_input")
    motion_input = Input(shape = input_shape_motion, name = "motion_input")

    speech = SpeechNetwork(speech_input)
    motion = MotionNetwork(motion_input)
    print(speech.shape, motion.shape)
    speech_motion = Lambda(CombineSpeechMotion)([speech, motion])
    model = Model(inputs = [speech_input, motion_input], outputs = [speech_motion])
    model.compile(optimizer='adam',loss=[NegativeCanonCorr],metrics=[NegativeCanonCorr])
    return model


    #Network for processing speech data
    speech_in = BatchNormalization()(speech_input)
    speech = Conv1D(24, 10, activation = 'relu', name = "speech_convolution")(speech_in)
    speech = BatchNormalization()(speech)
    speech = MaxPooling1D(pool_size = 10)(speech)
    print("Before LSTM", speech.shape)
    speech = Bidirectional(LSTM(units = 24, return_sequences = True, name = "speech_lstm"))(speech)
    speech = BatchNormalization()(speech)
    speech = GlobalAveragePooling1D(name = "speech_global_averager")(speech)
    speech = BatchNormalization()(speech)

    #Network for processing head motion data
    motion_in = BatchNormalization()(motion_input)
    motion = Conv1D(24, 10, activation = 'relu', name = "motion_convolution")(motion_in)
    motion = BatchNormalization()(motion)
    motion = MaxPooling1D(pool_size = 10)(motion)
    motion = Bidirectional(LSTM(units = 24, return_sequences = True, name = "motion_lstm"))(motion)
    motion = BatchNormalization()(motion)
    motion = GlobalAveragePooling1D(name = "motion_global_averager")(motion)
    motion = BatchNormalization()(motion) 
    #softmax_output = Dense(num_classes, activation = 'softmax', name = "final_predictor")(global_avg_output)
    speech_motion = Lambda(CombineSpeechMotion)([speech, motion])
    model = Model(inputs = [speech_input, motion_input], outputs = [speech_motion])
    #print(type(speech_motion), speech_motion.shape)
    model.compile(optimizer='adam',loss=[NegativeCanonCorr],metrics=[NegativeCanonCorr])
    return model


def linCCA(H1, H2, dim, rcov1, rcov2):

    N, d1 = H1.shape
    _, d2 = H2.shape

    # Remove mean.
    m1 = np.mean(H1, axis=0, keepdims=True)
    H1 = H1 - np.tile(m1, [N,1])

    m2 = np.mean(H2, axis=0, keepdims=True)
    H2 = H2 - np.tile(m2, [N,1])

    S11 = np.matmul(H1.transpose(), H1) / (N-1) + rcov1 * np.eye(d1)
    S22 = np.matmul(H2.transpose(), H2) / (N-1) + rcov2 * np.eye(d2)
    S12 = np.matmul(H1.transpose(), H2) / (N-1)

    #print("PRINTING LINEAR CCA")
    #print(np.sum(S11), np.sum(S22), np.sum(S12))
    E1, V1 = np.linalg.eig(S11)
    E2, V2 = np.linalg.eig(S22)

    # For numerical stability.
    idx1 = np.where(E1>eps_eig)[0]
    E1 = E1[idx1]
    V1 = V1[:, idx1]

    idx2 = np.where(E2>eps_eig)[0]
    E2 = E2[idx2]
    V2 = V2[:, idx2]
    #print(np.sum(E1), np.sum(V1), np.sum(E2), np.sum(V2))    

    K11 = np.matmul( np.matmul(V1, np.diag(np.reciprocal(np.sqrt(E1)))), V1.transpose())
    K22 = np.matmul( np.matmul(V2, np.diag(np.reciprocal(np.sqrt(E2)))), V2.transpose())
    T = np.matmul( np.matmul(K11, S12), K22)
    # print(T)
    U, E, V = np.linalg.svd(T, full_matrices=False)
    V = V.transpose()

    A = np.matmul(K11, U[:, 0:dim])
    B = np.matmul(K22, V[:, 0:dim])
    E = E[0:dim]

    #print(np.sum(K11), np.sum(K22), np.sum(T), np.mean(np.sqrt(E)))
    return A, B, m1, m2, E

rcov1 = 0.0001
rcov2 = 0.0001
dim = 4

arr_1 = np.random.rand(1000,32)
arr_2 = np.random.rand(1000,32)
ten_1 = tf.convert_to_tensor( arr_1 , dtype=tf.float32)
ten_2 = tf.convert_to_tensor( arr_2 , dtype=tf.float32)
_,_,_,_, lin_eigen = linCCA(arr_1, arr_2, dim, rcov1, rcov2)
cann_cor, _, _ = CanonCorr(ten_1, ten_2, 32, 32, 32, dim, rcov1, rcov2)

#print("Linear CCA", np.mean(np.sqrt(lin_eigen)) )
#print("Canonical Correlation", cann_cor.numpy() )
#sys.exit()

                    
DURATION = 600
NUM_CHANNELS_SPEECH = 13
NUM_CHANNELS_MOTION = 3

input_shape_speech = (DURATION, NUM_CHANNELS_SPEECH)
input_shape_motion = (DURATION, NUM_CHANNELS_MOTION)
output_size = 2

model = functional_CNN_pool_lstm_glob( input_shape_speech, input_shape_motion, output_size)
model.summary()
#sys.exit()

MOTION_COLS = ["X","Y","Z"]

velocity_train_files_location = "../Velocity/Train/VelocityFiles.txt"
velocity_valid_files_location = "../Velocity/Valid/VelocityFiles.txt"
velocity_test_files_location = "../Velocity/Test/VelocityFiles.txt"

mfcc_train_files_location = "../Velocity/Train/MFCCFiles.txt"
mfcc_valid_files_location = "../Velocity/Valid/MFCCFiles.txt"
mfcc_test_files_location = "../Velocity/Test/MFCCFiles.txt"

#ids_dictionary_train = IdAssigner(LinesToList(mfcc_train_files_location), 0, 1, duration = DURATION, shift = 60, l_chuck = 120, u_chuck = 120)
#labels_dictionary_train = IdAssignerMotion(LinesToList(velocity_train_files_location), 0, 1, MOTION_COLS, duration = DURATION, shift = 60, l_chuck = 120, u_chuck = 120)
#ids_dictionary_valid = IdAssigner(LinesToList(mfcc_valid_files_location), 0, 1, duration = DURATION, shift = 60, l_chuck = 120, u_chuck = 120)
#labels_dictionary_valid = IdAssignerMotion(LinesToList(velocity_valid_files_location), 0, 1, MOTION_COLS, duration = DURATION, shift = 60, l_chuck = 120, u_chuck = 120)

#with open(mfcc_train_files_location + ".bin", 'wb') as handle:
#    pickle.dump(ids_dictionary_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
#with open(velocity_train_files_location + ".bin", 'wb') as handle:
#    pickle.dump(labels_dictionary_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
#with open(mfcc_valid_files_location + ".bin", 'wb') as handle:
#    pickle.dump(ids_dictionary_valid, handle, protocol=pickle.HIGHEST_PROTOCOL)
#with open(velocity_valid_files_location + ".bin", 'wb') as handle:
#    pickle.dump(labels_dictionary_valid, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(mfcc_train_files_location + ".bin", 'rb') as handle:
    ids_dictionary_train = pickle.load(handle)
with open(velocity_train_files_location + ".bin", 'rb') as handle:
    labels_dictionary_train = pickle.load(handle)
with open(mfcc_valid_files_location + ".bin", 'rb') as handle:
    ids_dictionary_valid = pickle.load(handle)
with open(velocity_valid_files_location + ".bin", 'rb') as handle:
    labels_dictionary_valid = pickle.load(handle)

#train_generator = DataGenerator(ids_dictionary_train, labels_dictionary_train, dim = (DURATION,), n_classes = output_size)
#valid_generator = DataGenerator(ids_dictionary_valid, labels_dictionary_valid, dim = (DURATION,), n_classes = output_size)

train_generator = DataGenerator(ids_dictionary_train, labels_dictionary_train, dim = (DURATION,), n_channels_motion = NUM_CHANNELS_MOTION, n_classes = output_size)
valid_generator = DataGenerator(ids_dictionary_valid, labels_dictionary_valid, dim = (DURATION,), n_channels_motion = NUM_CHANNELS_MOTION, n_classes = output_size)

#print( len(labels_dictionary_train), len(labels_dictionary_valid) )
#for key, value in train_generator.VelocityDistribution().items():
#    print(key, value)
#for key, value in valid_generator.VelocityDistribution().items():
#    print(key, value)

#sys.exit()
model.fit_generator(generator = train_generator, epochs = 20, validation_data = valid_generator, use_multiprocessing = True, workers = 30)
sys.exit()

