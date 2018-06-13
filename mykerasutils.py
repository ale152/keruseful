# -*- coding: utf-8 -*-
"""
Created on Tue May 29 11:17:46 2018

@author: am14795
"""

import numpy as np
import matplotlib
import keras
from keras.callbacks import Callback
from keras.models import Model
from matplotlib import pyplot as plt

# %% Useful functions
def get_model_memory_usage(batch_size, model):
    '''Get the memory needed for a model to run'''
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)

    return gbytes


def get_max_batch_size(mem_avail, model, max_batch=None):
    '''Evaluate the maximum batch size that fits into memory'''
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    batch_size = mem_avail*(1024.0 ** 3)/(4.0*(shapes_mem_count + trainable_count + non_trainable_count))
    if max_batch is not None and batch_size > max_batch:
        batch_size = max_batch

    return int(batch_size)


# %% Useful callbacks 
class PlotResuls(Callback):
    '''This callback plots the losses and the prediction of the network during
    training'''
    def __init__(self, x_test, y_test, prediction_each=1, loss_each=1, 
                 savepred=None, saveloss=None): 
        self.x_test = x_test
        self.y_test = y_test
        self.prediction_each = prediction_each  # Show the prediciton each N epochs
        self.loss_each = loss_each  # Show the losses each N epochs
        self.savepred = savepred  # Save the prediction figure
        self.saveloss = saveloss  # Save the losses figure

    def on_train_begin(self, logs=None):
        # Create the figure and initialise the loss arrays
        self.fig_pred = plt.figure('Prediction')
        self.fig_loss = plt.figure('Losses')
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        # Store the losses at each epoch
        self.losses.append(logs['loss'])
        if 'val_loss' in logs:
            self.val_losses.append(logs['val_loss'])
        
        # Plot the loss
        if self.loss_each > 0 and (epoch+1) % self.loss_each == 0:
            plt.figure(self.fig_loss.number)
            plt.clf()
            plt.plot(self.losses, label='Loss')
            plt.plot(self.val_losses, label='Validation loss')
            if epoch > 100:
                plt.gca().set_xscale('log')
            plt.legend()
            plt.pause(0.01)
            if self.saveloss is not None:
                plt.savefig('%s.png' % self.saveloss)
        
        # Plot the prediction
        if self.prediction_each > 0 and (epoch+1) % self.prediction_each == 0:
            plt.figure(self.fig_pred.number)
            plt.clf()
            y_pred = self.model.predict(self.x_test[None, ])[0, ]
            
            plt.subplot(121)
            if len(self.y_test.shape) == 1:
                plt.plot(self.y_test)    
            elif len(self.y_test.shape) == 2:
                plt.imshow(self.y_test)
            plt.title('Original')
            
            plt.subplot(122)
            if len(y_pred.shape) == 1:
                plt.plot(y_pred)
            elif len(y_pred.shape) == 2:
                plt.imshow(y_pred)
            plt.title('Prediction')
            
            plt.pause(0.1)
            if self.savepred is not None:
                plt.savefig('%s_%d.png' % (self.savepred, epoch))

   
class ShowEveryLayer(Callback):
    '''This callbacks uses the function show_model_output'''
    def __init__(self, model, x_target, layers=None, show_each=1, verbose=0, savefig=None):
        self.model = model
        self.x_target = x_target
        self.layers = layers
        self.show_each = show_each
        self.verbose = verbose
        self.savefig = savefig
        
    def on_epoch_end(self, epoch, logs=None):
        # Plot the intermediate layers
        if self.show_each > 0 and (epoch+1) % self.show_each == 0:
            show_intermediate_output(self.model, self.x_target, self.layers,
                                     self.verbose, self.savefig)

			
def show_intermediate_output(model, x_target, layers=None, verbose=0, savefig=None):
    '''Show the output of the intermediate layers of the model'''
    # Enumare the layers
    if layers is None:
        layerate = enumerate(model.layers)
    else:
        def layerator():
            for layer in layers:
                if isinstance(layer, str):
                    bf = next((li, lay) for li, lay in enumerate(model.layers) if lay.name == layer)
                    yield bf[0], bf[1]
                elif isinstance(layer, int):
                    yield layer, model.layers[layer]
        layerate = layerator()
        
    # Loop over the layers
    for li, layer in layerate:
        # Get the output of the layer
        submodel = Model(inputs=model.input, outputs=layer.output)
        subout = submodel.predict(x_target[None, ])[0, ]

        # Save layers dimension
        real_dims = subout.shape
        
        # Plot the outputs
        if verbose == 1:
            print('[%d] %s: %s' % (li, layer.name, str(real_dims)))
        plt.figure('#%d, %s' % (li, layer.name))
        plt.clf()
        
        # One dimensional output or two dimensional with less than 3 elements
        if len(subout.shape) == 1:
            # If the layer is Dense, show the input, weights and output
            if isinstance(layer, keras.layers.Dense):
                # Find the input of this layer (output of previous layer)
                prevmodel = Model(inputs=model.input, outputs=model.layers[li-1].output)
                prevout = prevmodel.predict(x_target[None, ])[0, ]
                plt.subplot(1, 5, 1)
                x = np.arange(len(prevout))
                plt.plot(prevout, x)
                plt.title('Input')
                
                # Show the weight matrix
                weights, biases = layer.get_weights()
                plt.subplot(1, 5, (2, 3))
                plt.imshow(weights, cmap='jet')
                plt.axis('tight')
                plt.title('Weights')
                plt.colorbar()
                
                # Plot biases
                plt.subplot(1, 5, 4)
                x = np.arange(len(biases))
                plt.plot(biases, x)
                plt.axis('tight')
                plt.title('Biases')
                
                # Plot output
                plt.subplot(1, 5, 5)
                x = np.arange(len(subout))
                plt.plot(subout, x)
                
                plt.tight_layout()
            else:
                # If it's not Dense, just plot the output
                plt.plot(subout)  
            
        elif len(subout.shape) == 2:
            # Image plot
            plt.imshow(subout, cmap='jet')
            plt.axis('tight')
            plt.colorbar()
        elif len(subout.shape) == 3:
            # Squeeze all the images on a grid
            subout = squeeze_imgseq_to_grid(subout)
            # Set NaN margins to black
            cmap = matplotlib.cm.jet
            cmap.set_bad('k')
            plt.imshow(subout, cmap='jet')
            plt.axis('tight')
            plt.colorbar()
        elif len(subout.shape) == 4:
            # First, squeeze each image in the 3rd dimension in a grid and
            # make a sequence of grids
            init = True
            Nseq = subout.shape[3]
            for i in range(Nseq):
                grid = squeeze_imgseq_to_grid(subout[:, :, :, i])
                if init:
                    init = False
                    seq_of_grid = np.zeros((grid.shape[0], grid.shape[1], Nseq))
                    
                seq_of_grid[:, :, i] = grid
                
            # Second, convert the sequence of grids into a grid
            grid_of_grids = squeeze_imgseq_to_grid(seq_of_grid)
            
            # Set NaN margins to black
            cmap = matplotlib.cm.jet
            cmap.set_bad('k')
            plt.imshow(grid_of_grids, cmap='jet')
            plt.axis('tight')
            plt.colorbar()
            
        plt.title('Layer: %s, size: %s' % (layer.name, str(real_dims)))
        plt.pause(0.1)
        if savefig is not None:
            plt.savefig('%s_%d_%s.png' % (savefig, li, layer.name))

       
def squeeze_imgseq_to_grid(imgseq, ratio=16/9, margin_size=1, margin_value=np.nan):
    '''Take a 3D matrix containing 'Nimg' images in the form of (x, y, Nimg) 
    and output a single image containing the 'Nimg' images disposed on a 
    2D grid'''
    # Pad the output with NaN according to margin
    Ny = int(np.ceil(np.sqrt(imgseq.shape[2]/ratio)))
    Nx = int(np.ceil(np.sqrt(imgseq.shape[2]*ratio)))
    N3d = Nx*Ny
    padded = np.zeros((imgseq.shape[0]+margin_size*2, 
                       imgseq.shape[1]+margin_size*2, 
                       N3d))
    for i in range(imgseq.shape[2]):
        padded[:, :, i] = np.pad(imgseq[:, :, i], margin_size, 
              'constant', constant_values=margin_value)
        
    # Reshape the output into a 2D grid
    dim = padded.shape
    imgseq = np.reshape(np.transpose(padded, (0, 2, 1)), (dim[0], Ny, Nx, dim[1]), order='F')
    imgseq = np.reshape(np.transpose(imgseq, (0, 1, 3, 2)), (dim[0]*Ny, dim[1]*Nx), order='F')
    
    return imgseq


# %% Test
if __name__ == '__main__':
    import keras
    from keras.datasets import mnist
    from keras.models import Model, Input
    from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Reshape
    
    # Download mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[:300, ]
    y_train = y_train[:300, ]
    x_test = x_test[:300, ]
    y_test = y_test[:300, ]
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    # Simple Convnet
    inputs = Input(shape=(28, 28, ), name='input')
    net = Reshape((28, 28, 1))(inputs)
    net = Conv2D(32, kernel_size=(3, 3))(net)
    net = MaxPooling2D(pool_size=(2, 2))(net)
    net = Conv2D(32, kernel_size=(3, 3))(net)
    net = MaxPooling2D(pool_size=(2, 2))(net)
    net = Conv2D(32, kernel_size=(3, 3))(net)
    net = MaxPooling2D(pool_size=(2, 2))(net)
    net = Flatten()(net)
    net = Dense(128, activation='relu')(net)
    outputs = Dense(10, activation='softmax')(net)
    
    # Compile the model
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta())

    # Define the callbacks
    plot_results = PlotResuls(x_test[0, ], y_test[0, ])
    show_every_layer = ShowEveryLayer(model, x_test[0, ], show_each=2, layers=[3, 'input'], savefig='MyOutput')
    model.fit(x_train, y_train, validation_data=(x_test, y_test),
              epochs=10, callbacks=[plot_results, show_every_layer])

    
    #show_intermediate_output(model, x_train[0, ...])
