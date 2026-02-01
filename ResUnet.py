import numpy as np
from keras.backend import int_shape
from keras.models import Model
from keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, UpSampling2D, UpSampling3D, Add, BatchNormalization, Input, Activation, Lambda, Concatenate

def res_unet(filter_root, depth, n_class=2, input_size=(256,256,1), activation ='relu', batch_norm = True, final_activation = 'softmax'):
    #build the model
    inputs = Input(input_size)
    x = inputs

    long_connection_store = {}
    if len(input_size) == 3:
        Conv = Conv2D
        MaxPooling = MaxPooling2D
        UpSampling = UpSampling2D
    
    if len(input_size) == 4:
        Conv = Conv3D
        MaxPooling = MaxPooling3D
        UpSampling = UpSampling3D
    
    #Down sampling
    for i in range(depth):
        out_channel = 2**i * filter_root

        res = Conv(out_channel, kernel_size = 1, padding = 'same', use_bias = False, name = "Identity{}_1".format(i))(x)

        conv1 = Conv(out_channel, kernel_size = 3, padding = 'same', name= "Conv{}_1".format(i))(x)
        if batch_norm:
            conv1 = BatchNormalization(name="BN{}_1".format(i))(conv1)
        act1 = Activation(activation, name= "Act{}_1".format(i))(conv1)

        conv2 = Conv(out_channel, kernel_size=3, padding = 'same', name = "Conv{}_2".format(i))(act1)
        if batch_norm:
            conv2 = BatchNormalization(name="BN{}_2".format(i))(conv2)
        
        resconnection = Add(name="Add{}_1".format(i))([res, conv2])

        act2 = Activation(activation, name="Act{}_2".format(i))(resconnection)

        if i < depth - 1:
            long_connection_store[str(i)] = act2
            x = MaxPooling(padding='same',name="MaxPooling{}_1".format(i))(act2)
        else:
            x = act2

    #Upsampling
    for i in range(depth-2, -1, -1):
        out_channel = 2**(i) * filter_root

        long_connection = long_connection_store[str(i)]

        up1 = UpSampling(name="UpSampling{}_1".format(i))(x)
        up_conv1 = Conv(out_channel, 2, activation = 'relu', padding = 'same', name = 'upConv{}_1'.format(i))(up1)

        up_conc = Concatenate(axis=-1, name="upConcatenate{}_1".format(i))([up_conv1, long_connection])

        up_conv2 = Conv(out_channel, 3, padding = 'same', name = "upConv{}_1_".format(i))(up_conc)
        if batch_norm:
            up_conv2 = BatchNormalization(name="upBN{}_2".format(i))(up_conv2)
        
        res = Conv(out_channel, kernel_size = 1, padding = 'same', use_bias = False, name = 'UpIdentity{}_1'.format(i))(up_conv2)

        resconnection = Add(name= "upAdd{}_1".format(i))([res,up_conv2])

        x = Activation(activation, name = "upAct{}_2".format(i))(resconnection)
    
    output = Conv(1, 1, padding = 'same', activation = final_activation, name = 'output')

    return Model(inputs, outputs = output, name= 'Res-Unet')

model = res_unet(64, 5, n_class=2, input_size=(512, 512, 3), activation = 'relu', batch_norm=True, final_activation='softmax')
model.summary()