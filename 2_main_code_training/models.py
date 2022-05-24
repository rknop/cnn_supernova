## modify this script to change the model.
### Add models with a new index.

import copy
from tensorflow.keras import layers, models, optimizers, callbacks  # or tensorflow.keras as keras
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K

### Import the modules for resnet50
from resnet50 import *
from resnet18 import *

### Defining all the models tried in the study


def f_model_prototype(shape,**model_dict):
    '''
    General prototype for layered CNNs
    
    Structure:
    For different conv layers:
        - Conv 1
        - Conv 2
        - Pooling
        - Dropout
    - Flatten
    - Dropout 
    - Dense 
    - Dense -> 1
    
    '''
   
    activ='relu' # activation
    inputs = layers.Input(shape=shape)
    h = inputs
    # Convolutional layers
    conv_sizes=model_dict['conv_size_list'] # Eg. [10,10,10]
    
    ### Striding
    if model_dict['strides'] == 1:
        stride_lst = [1] * len(conv_sizes) # Default stride is 1 for each convolution.
    else : 
        stride_lst=model_dict['strides']
    
    conv_args = dict(kernel_size=model_dict['kernel_size'], activation=activ, padding='same')
    
    for conv_size,strd in zip(conv_sizes,stride_lst):
        h = layers.Conv2D(conv_size, strides=strd, **conv_args)(h)
        h = layers.BatchNormalization(epsilon=1e-5, momentum=0.99)(h)
        
        if model_dict['double_conv']: 
            h = layers.Conv2D(conv_size,strides=strd, **conv_args)(h)
            h = layers.BatchNormalization(epsilon=1e-5, momentum=0.99)(h)
        
        if not model_dict['no_pool']: h = layers.MaxPooling2D(pool_size=model_dict['pool_size'])(h)
        ## inner_dropout is None or a float
        if model_dict['inner_dropout']!=None: h = layers.Dropout(rate=model_dict['inner_dropout'])(h)
    h = layers.Flatten()(h)
    
    # Fully connected  layers
    if model_dict['outer_dropout']!=None: h = layers.Dropout(rate=model_dict['outer_dropout'])(h)
    
    h = layers.Dense(model_dict['dense_size'], activation=activ)(h)
    h = layers.BatchNormalization(epsilon=1e-5, momentum=0.99)(h)
    
    # Ouptut layer
    outputs = layers.Dense(1, activation=model_dict['final_activation'])(h)    
    return outputs,inputs
    
def f_define_model(config_dict,name='1'):
    '''
    Function that defines the model and compiles it. 
    Reads in a dictionary with parameters for CNN model prototype and returns a keral model
    '''
    ### Extract info from the config_dict
    shape=config_dict['model']['input_shape']
    loss_fn=config_dict['training']['loss']
    metrics=config_dict['training']['metrics']
    
    resnet=False ### Variable storing whether the models is resnet or not. This is needed for specifying the loss function.    
    custom_model=False ### Variable storing whether the models is a layer-by-layer build code (not using the protytype function).    

    model_list = {}

    model_list[1] = {'conv_size_list': [80, 80, 80],
                     'kernel_size': (3, 3),
                     'no_pool': False,
                     'pool_size': (2, 2),
                     'strides': 1,
                     'learn_rate': 0.00002,
                     'inner_dropout': None,
                     'outer_dropout': 0.3,
                     'dense_size': 51,
                     'final_activation': 'sigmoid',
                     'double_conv': False }
    model_list[2] = {'conv_size_list': [80, 80],
                     'kernel_size': (4, 4),
                     'no_pool': False,
                     'pool_size': (3, 3),
                     'strides': 1,
                     'learn_rate': 0.00002,
                     'inner_dropout': None,
                     'outer_dropout': 0.3,
                     'dense_size': 51,
                     'final_activation': 'sigmoid',
                     'double_conv': True }
    model_list[3] = {'conv_size_list': [120,120] ,
                     'kernel_size': (4,4) ,
                     'no_pool': False ,
                     'pool_size': (3,3) ,
                     'strides': 1 ,
                     'learn_rate': 0.00002 ,
                     'inner_dropout': None ,
                     'outer_dropout': 0.3 ,
                     'dense_size': 51 ,
                     'final_activation': 'sigmoid' ,
                     'double_conv': True }
    model_list[4] = {'conv_size_list': [40,60,80],
                     'kernel_size': (6,6),
                     'no_pool': True,
                     'pool_size': (2,2),
                     'strides': [2,2,1],
                     'learn_rate': 0.00002,
                     'inner_dropout': 0.1,
                     'outer_dropout': 0.3,
                     'dense_size': 51,
                     'final_activation': 'sigmoid',
                     'double_conv': False }

    # models 5-8 : start with 3 as a base, vary pool size
    # Model 3 was the best of these (batch_size 128)
    
    newmods = range(5, 9)
    poolsizes = ( 2, 3, 5, 6 )
    for i in newmods:
        model_list[i] = copy.deepcopy( model_list[3] )
        model_list[i]['pool_size'] = ( poolsizes[i-newmods[0]], poolsizes[i-newmods[0]] )

    # newmods 9-14 : start with 3 as base, vary kernel size
    # Model 3 was the best of these, 10 was second (batch_size 128)
    # Bigger sizes get really bad
    newmods = range(9, 15)
    kernelsizes = ( 2, 3, 5, 6, 7, 8 )
    for i in newmods:
        model_list[i] = copy.deepcopy( model_list[3] )
        model_list[i]['kernel_size'] = ( kernelsizes[i-newmods[0]], kernelsizes[i-newmods[0]] )

    # newmods 15-18 : start with 3 as base, futz with inner dropouts
    # 15 was best, not a lot better than 3
    
    newmods = range(15, 19)
    innerdropouts = ( 0.1, 0.2, 0.3, 0.4 )
    for i in newmods:
        model_list[i] = copy.deepcopy( model_list[3] )
        model_list[i]['inner_dropout'] = innerdropouts[ i - newmods[0] ]

    # newmods 19-22 : start with 15 as base, futz with outer dropouts
    # Doesn't make a lot of difference  15, 19, 20 are best, maybe 19 marginally better
    newmods = range( 19, 23 )
    outerdropouts = ( 0.1, 0.2, 0.4, 0.5 )
    for i in newmods:
        model_list[i] = copy.deepcopy( model_list[15] )
        model_list[i]['outer_dropout'] = outerdropouts[ i - newmods[0] ]

    # newmods 23-25 : start with 19 as base, futz with conv sizes and strides
    # None of these were as good as 19
    newmods = range( 23, 26 )
    convsizes = ( [ 60, 100, 60 ], [40, 60, 40], [40, 60, 80] )
    strides = ( [ 1, 2, 1], [ 1, 2, 1 ], [ 1, 2, 1 ] )
    for i in newmods:
        model_list[i] = copy.deepcopy( model_list[19] )
        model_list[i]['conv_size_list'] = convsizes[ i - newmods[0] ]
        model_list[i]['strides'] = strides[ i - newmods[0] ]
        model_list[i]['no_pool'] = True
        
    ############################################
    ### Add more models above
    ############################################
    ####### Compile model ######################
    ############################################

    name = int(name)
    if not name in model_list:
        raise ValueError( f'Unknown model {name}' )
    model_par_dict = model_list[name]
        
    if resnet:
        print("resnet model name",name)
        opt,loss_fn=optimizers.Adam(lr=learn_rate),'sparse_categorical_crossentropy'
    else : ## For non resnet models 
        if not custom_model:  ### For non-custom models, use prototype function
            outputs,inputs=f_model_prototype(shape,**model_par_dict)
            learn_rate=model_par_dict['learn_rate']    
        model = models.Model(inputs, outputs)
        opt=optimizers.Adam(lr=learn_rate)
    
    model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)
    #print("model %s"%name)
    
    return model

