import os

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

import pandas as pd
import numpy as np
import sys
import keras

from keras.models import Sequential, Model
from keras.layers import SeparableConv1D, AveragePooling1D, CuDNNLSTM, Concatenate, Lambda, Reshape, Input, Bidirectional, TimeDistributed, GlobalAveragePooling1D, SpatialDropout1D, Conv1D, BatchNormalization, MaxPooling1D, Dense, Activation, Flatten, LSTM, Dropout
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback


"""
files - List of files to use for generating Ids
lp - Starting point of the data frames
up - End point of the data frames
duration - Number of frames to constitute one data point
l_chuck - Number of frames to chuck at the beginning
u_chuck - Number of frames to chuck at the end
"""
def IdAssigner(files, lp, up, duration = 120, l_chuck = 0, u_chuck = 0):
    counter = 1
    id_dictionary = {}
    for fi in files:
        df = pd.read_csv(fi)
        num_frames = len(df)
        clb = l_chuck + int(lp * num_frames) #Current lower bound
        cub = clb + duration
        
        while num_frames - int( (1 - up) * num_frames ) - u_chuck >= cub:
            id_dictionary[counter] = (fi, clb, cub)
            counter += 1
            clb = cub
            cub += duration
            
    return id_dictionary

def LabelIdAssigner(files, lp, up, duration = 120, l_chuck = 0, u_chuck = 0):
    counter = 1
    id_dictionary = {}
    for fi in files:
        print(fi)
        df = pd.read_csv(fi)
        num_frames = len(df)
        clb = l_chuck + int(lp * num_frames) #Current lower bound
        cub = clb + duration
        
        while num_frames - int( (1 - up) * num_frames ) - u_chuck >= cub:
            if VelocityGrader(df.iloc[clb:cub, :]) == -1:
                counter += 1
                clb = cub
                cub += duration
                continue
            id_dictionary[counter] = (fi, clb, cub)
            counter += 1
            clb = cub
            cub += duration
            
    return id_dictionary    

def VelocityGrader(df):
    velocity_distribution = {}
     
    faster_bools = df["faster_vel"].astype(int)
    
    if sum(faster_bools) < int( 0.1 * len(faster_bools) ):
        return 0
        velocity_distribution[0] = velocity_distribution.get(0, 0) + 1
    
    elif sum(faster_bools) < int( 0.2 * len(faster_bools) ):
        return 0
        velocity_distribution[1] = velocity_distribution.get(1, 0) + 1
        
    elif sum(faster_bools) < int( 0.3 * len(faster_bools) ):
        return 0
        velocity_distribution[2] = velocity_distribution.get(2, 0) + 1
        
    elif sum(faster_bools) < int( 0.4 * len(faster_bools) ):
        return 0
        velocity_distribution[3] = velocity_distribution.get(3, 0) + 1
        
    elif sum(faster_bools) < int( 0.5 * len(faster_bools) ):
        return 0
        velocity_distribution[4] = velocity_distribution.get(4, 0) + 1
        
    elif sum(faster_bools) < int( 0.6 * len(faster_bools) ):
        return 1
        velocity_distribution[5] = velocity_distribution.get(5, 0) + 1
        
    elif sum(faster_bools) < int( 0.7 * len(faster_bools) ):
        return 1
        velocity_distribution[6] = velocity_distribution.get(6, 0) + 1
        
    elif sum(faster_bools) < int( 0.8 * len(faster_bools) ):
        return 1
        velocity_distribution[7] = velocity_distribution.get(7, 0) + 1
        
    elif sum(faster_bools) < int( 0.9 * len(faster_bools) ):
        return 1
        velocity_distribution[8] = velocity_distribution.get(8, 0) + 1                                                                                                                            

    else:
        return 1
        velocity_distribution[9] = velocity_distribution.get(9, 0) + 1    

def LinesToList(location):
    with open(location) as f:
        all_lines = f.read()
    lines = all_lines.split("\n")
    return lines[:-1]
    

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, ids_dictionary, labels_dictionary, batch_size=32, dim=(120,), n_channels=13,
                 n_classes=3, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels_dictionary
        self.list_IDs = list(labels_dictionary.keys())
        self.n_channels = n_channels
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
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = self.list_IDs #In turn, list_IDs also get shuffled
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self.__GetOneDataPoint(ID)#np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.__GetLabel(ID)

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        
    def __GetOneDataPoint(self, ID):
        location, ll, ul = self.ids_dictionary[ID]
        df = pd.read_csv(location).iloc[ll:ul, :]
        return np.array(df)
        
    def __GetLabel(self, ID):
        location, ll, ul = self.labels[ID]                            
        df = pd.read_csv(location).iloc[ll:ul, :]
        return self.__VelocityGrader(df)
        
    def __VelocityGrader(self, df):
        faster_bools = df["faster_vel"].astype(int)
        if sum(faster_bools)  < int ( len(faster_bools) * 0.3 ):
            return 0
        if sum(faster_bools) > int ( len(faster_bools) * 0.7 ):
            return 2
        return 1
        
#        if sum(faster_bools) * 2 < len(faster_bools):
#            return 0
            
#        return 1

            


def functional_CNN_pool_lstm_glob( input_shape, num_classes ):
    motion_input = Input(shape = input_shape, name = "input")
    conv_output = Conv1D(24, 10, activation = 'relu', name = "convolution")(motion_input)#Conv1D(24, 10, activation = 'relu', name = "convolution")(motion_input)
    bn = BatchNormalization()(conv_output) #Trial
    lap = MaxPooling1D(pool_size = 10)(bn) #Trial - conv_output
    lstm = Bidirectional(CuDNNLSTM(units = 24, return_sequences = True, name = "lstm"))(lap)#CuDNNLSTM(units = 24, return_sequences = True, name = "lstm")(lap)
    bn_2 = BatchNormalization()(lstm) #Trial
    #print(lap.shape)
    global_avg_output = GlobalAveragePooling1D(name = "global_averager")(bn_2) #Trial - lstm
    #global_avg_output = Flatten()(lap)
    softmax_output = Dense(num_classes, activation = 'softmax', name = "final_predictor")(global_avg_output)
    model = Model(inputs = [motion_input], outputs = [softmax_output])
    model.compile(optimizer='adam',loss=['categorical_crossentropy'],metrics=['categorical_accuracy'])
    return model
                    
DURATION = 120
NUM_CHANNELS = 13

input_shape = (DURATION, NUM_CHANNELS)
output_size = 3

model = functional_CNN_pool_lstm_glob( input_shape, output_size)
model.summary()
#sys.exit()

ids_dictionary_train = IdAssigner(LinesToList("/home2/data/Sanjeev/Velocity/MFCCFiles.txt"), 0, 0.75, duration = DURATION)
labels_dictionary_train = LabelIdAssigner(LinesToList("/home2/data/Sanjeev/Velocity/VelocityFiles.txt"), 0, 0.75, duration = DURATION)
ids_dictionary_valid = IdAssigner(LinesToList("/home2/data/Sanjeev/Velocity/MFCCFiles.txt"), 0.75, 1, duration = DURATION)
labels_dictionary_valid = LabelIdAssigner(LinesToList("/home2/data/Sanjeev/Velocity/VelocityFiles.txt"), 0.75, 1, duration = DURATION)

train_generator = DataGenerator(ids_dictionary_train, labels_dictionary_train, dim = (DURATION,))
valid_generator = DataGenerator(ids_dictionary_valid, labels_dictionary_valid, dim = (DURATION,))
model.fit_generator(generator = train_generator, epochs = 20, validation_data = valid_generator, use_multiprocessing = True, workers = 240)
sys.exit()
dg_obj = DataGenerator(ids_dictionary, labels_dictionary, dim = (DURATION,))
x, y = dg_obj[1]
print(x.shape, y.shape)
sys.exit()    

num_data_points = len(id_dictionary)
ids = list(id_dictionary.keys())
np.random.shuffle(ids)
print(id_dictionary.keys())
print(ids)
print(num_data_points)    
