import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input,merge
from keras import initializations
from keras.utils import visualize_util
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import linear
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten, Dense, Activation, Reshape, Lambda
from keras.layers.convolutional import Convolution2D, Deconvolution2D, UpSampling2D, MaxPooling2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.layers.noise import GaussianNoise
from keras.regularizers import *

def conv2D_init(shape, name=None,dim_ordering=None):
    return initializations.normal(shape, scale=0.02, name=name)


def wasserstein(y_true, y_pred):

    # return K.mean(y_true * y_pred) / K.mean(y_true)
    return K.mean(y_true * y_pred)


def visualize_model(model):

    model.summary()
    visualize_util.plot(model,
                        to_file='../../figures/%s.png' % model.name,
                        show_shapes=True,
                        show_layer_names=True)



def generator_upsampling_mnistM(noise_dim, img_source_dim,img_dest_dim, bn_mode,deterministic,inject_noise,wd, model_name="generator_upsampling", dset="mnistM"):
    """DCGAN generator based on Upsampling and Conv2D

    Args:
        noise_dim: Dimension of the noise input
        img_dim: dimension of the image output
        bn_mode: keras batchnorm mode
        model_name: model name (default: {"generator_upsampling"})
        dset: dataset (default: {"mnist"})

    Returns:
        keras model
    """
    s = img_source_dim[1]
    f = 512
#    shp = np.expand_dims(img_dim[1:],1) # to make shp= (None, 1, 28, 28)  but is not working
    start_dim = int(s / 4)
    nb_upconv = 2
    nb_filters = 64
    if K.image_dim_ordering() == "th":
        bn_axis = 1
        input_channels = img_source_dim[0]
        output_channels = img_dest_dim[0]
        reshape_shape = (input_channels, s, s)
        shp=reshape_shape

    else:
        bn_axis = -1
        input_channels = img_source_dim[-1]
        output_channels = img_dest_dim[-1]
        reshape_shape = (s, s, input_channels)
        shp=reshape_shape 
    gen_noise_input = Input(shape=noise_dim, name="generator_input")
    gen_image_input = Input(shape=shp, name="generator_image_input")
    #import code
    #code.interact(local=locals()
    # Noise input and reshaping
    x = Dense(s*s*input_channels, input_dim=noise_dim,W_regularizer=l2(wd))(gen_noise_input)
    x = Reshape(reshape_shape)(x)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)

    x = Activation("relu")(x)
    if deterministic: #here I link or not link the noise vector to the whole network
        g = gen_image_input
    else:
        g = merge([gen_image_input, x], mode='concat',concat_axis=1) # because of concat_axis=1, will it work on tensorflow too? 

    if inject_noise:
        g = GaussianNoise( sigma=0.02 )(g)
    g_64feats = Convolution2D(nb_filters, 3, 3, border_mode='same', init=conv2D_init,W_regularizer=l2(wd))(g) #convolved by 3x3 filter to get 64x55x35
    g_64feats = Activation('relu')(g_64feats)
    if inject_noise:
        g_64feats = GaussianNoise( sigma=0.02 )(g_64feats)

    H0 = Convolution2D(nb_filters, 3, 3, border_mode='same', init=conv2D_init,W_regularizer=l2(wd))(g_64feats)
    H0 = BatchNormalization(mode=bn_mode,axis=1)(H0)  
    H0 = Activation('relu')(H0)
    if inject_noise:
        H0 = GaussianNoise( sigma=0.02 )(H0)
    H0 = Convolution2D(nb_filters, 3, 3, border_mode='same', init=conv2D_init,W_regularizer=l2(wd))(H0)
    H0 = BatchNormalization(mode=bn_mode,axis=1)(H0)

    H0 = merge([H0, g_64feats], mode='sum')
    H0 = Activation('relu')(H0)
    if inject_noise:
        H0 = GaussianNoise( sigma=0.02 )(H0)

    H1 = Convolution2D(nb_filters, 3, 3, border_mode='same', init=conv2D_init,W_regularizer=l2(wd))(H0)
    H1 = BatchNormalization(mode=bn_mode,axis=1)(H1)  
    H1 = Activation('relu')(H1)
    if inject_noise:
        H1 = GaussianNoise( sigma=0.02 )(H1)
    H1 = Convolution2D(nb_filters, 3, 3, border_mode='same', init=conv2D_init,W_regularizer=l2(wd))(H1)
    H1 = BatchNormalization(mode=bn_mode,axis=1)(H1)

    H1 = merge([H0, H1], mode='sum')
    H1 = Activation('relu')(H1)
    if inject_noise:
        H1 = GaussianNoise( sigma=0.02 )(H1)

    H2 = Convolution2D(nb_filters, 3, 3, border_mode='same', init=conv2D_init,W_regularizer=l2(wd))(H1)
    H2 = BatchNormalization(mode=bn_mode,axis=1)(H2)  
    H2 = Activation('relu')(H2)
    if inject_noise:
        H2 = GaussianNoise( sigma=0.02 )(H2)
    H2 = Convolution2D(nb_filters, 3, 3, border_mode='same', init=conv2D_init,W_regularizer=l2(wd))(H2)
    H2 = BatchNormalization(mode=bn_mode,axis=1)(H2)
    H2 = merge([H1, H2], mode='sum')
    H2 = Activation('relu')(H2)

    if inject_noise:
        H2 = GaussianNoise( sigma=0.02 )(H2)
    H3 = Convolution2D(nb_filters, 3, 3, border_mode='same', init=conv2D_init,W_regularizer=l2(wd))(H2)
    H3 = BatchNormalization(mode=bn_mode,axis=1)(H3)  
    H3 = Activation('relu')(H3)
    if inject_noise:
        H3 = GaussianNoise( sigma=0.02 )(H3)
    H3 = Convolution2D(nb_filters, 3, 3, border_mode='same', init=conv2D_init,W_regularizer=l2(wd))(H3)
    H3 = BatchNormalization(mode=bn_mode,axis=1)(H3)

    H3 = merge([H2, H3], mode='sum')
    H11 = Activation('relu')(H3)

    # Last Conv to get the output image
    if inject_noise:
        H11 = GaussianNoise( sigma=0.02 )(H11)
    H11 = Convolution2D(output_channels, 1, 1,name="gen_conv2d_final", border_mode='same', init=conv2D_init,W_regularizer=l2(wd))(H11)
    g_V = Activation('tanh')(H11)

    generator_model = Model(input=[gen_noise_input,gen_image_input], output=[g_V], name=model_name)
    visualize_model(generator_model)

    return generator_model



def generator_upsampling(noise_dim, img_dim, bn_mode, model_name="generator_upsampling", dset="mnist"):
    """DCGAN generator based on Upsampling and Conv2D

    Args:
        noise_dim: Dimension of the noise input
        img_dim: dimension of the image output
        bn_mode: keras batchnorm mode
        model_name: model name (default: {"generator_upsampling"})
        dset: dataset (default: {"mnist"})

    Returns:
        keras model
    """

    s = img_dim[1]
    f = 512

    if dset == "mnist":
        start_dim = int(s / 4)
        nb_upconv = 2
    else:
        start_dim = int(s / 16)
        nb_upconv = 4

    if K.image_dim_ordering() == "th":
        bn_axis = 1
        reshape_shape = (f, start_dim, start_dim)
        output_channels = img_dim[0]
    else:
        reshape_shape = (start_dim, start_dim, f)
        bn_axis = -1
        output_channels = img_dim[-1]

    gen_noise_input = Input(shape=noise_dim, name="generator_input")

    # Noise input and reshaping
    x = Dense(f * start_dim * start_dim, input_dim=noise_dim)(gen_noise_input)
    x = Reshape(reshape_shape)(x)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    x = Activation("relu")(x)

    # Upscaling blocks: Upsampling2D->Conv2D->ReLU->BN->Conv2D->ReLU
    for i in range(nb_upconv):
        x = UpSampling2D(size=(2, 2))(x)
        nb_filters = int(f / (2 ** (i + 1)))
        x = Convolution2D(nb_filters, 3, 3, border_mode="same", init=conv2D_init)(x)
        x = BatchNormalization(mode=bn_mode, axis=1)(x)
        x = Activation("relu")(x)
        x = Convolution2D(nb_filters, 3, 3, border_mode="same", init=conv2D_init)(x)
        x = Activation("relu")(x)

    # Last Conv to get the output image
    x = Convolution2D(output_channels, 3, 3, name="gen_conv2d_final",
                      border_mode="same", activation='tanh', init=conv2D_init)(x)

    generator_model = Model(input=[gen_noise_input], output=[x], name=model_name)
    visualize_model(generator_model)
    return generator_model

#def mdb_layer():
#    if use_mbd:
#        x = Flatten()(x)

#    def minb_disc(x):
#        diffs = K.expand_dims(x, 3) - K.expand_dims(K.permute_dimensions(x, [1, 2, 0]), 0)
#        abs_diffs = K.sum(K.abs(diffs), 2)
#        x = K.sum(K.exp(-abs_diffs), 2)

#        return x

#    def lambda_output(input_shape):
#        return input_shape[:2]

#    num_kernels = 100
#    dim_per_kernel = 5

#    M = Dense(num_kernels * dim_per_kernel, bias=False, activation=None)
#    MBD = Lambda(minb_disc, output_shape=lambda_output)

#    if use_mbd:
#        x_mbd = M(x)
#        x_mbd = Reshape((num_kernels, dim_per_kernel))(x_mbd)
#        x_mbd = MBD(x_mbd)
#       x = merge([x, x_mbd], mode='concat')
#        x = Dense(1, name="disc_dense_1")(x)

def discriminator(img_dim, bn_mode,model,wd,monsterClass,n_classes, model_name="discriminator",use_mbd=False):
    """DCGAN discriminator

    Args:
        img_dim: dimension of the image output
        bn_mode: keras batchnorm mode
        model_name: model name (default: {"generator_deconv"})

    Returns:
        keras model
    """

    if K.image_dim_ordering() == "th":
        bn_axis = 1
        min_s = min(img_dim[1:])
    else:
        bn_axis = -1
        min_s = min(img_dim[:-1])

    disc_input = Input(shape=img_dim, name="discriminator_input")

    # Get the list of number of conv filters
    # (first layer starts with 64), filters are subsequently doubled
    nb_conv =int(np.floor(np.log(min_s // 4) / np.log(2)))
    list_f = [64 * min(8, (2 ** i)) for i in range(nb_conv)]

    # First conv with 2x2 strides
    x = Convolution2D(list_f[0], 3, 3, subsample=(2, 2), name="disc_conv2d_1",
                      border_mode="same", bias=False, init=conv2D_init,W_regularizer=l2(wd))(disc_input)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    x = LeakyReLU(0.2)(x)

    # Conv blocks: Conv2D(2x2 strides)->BN->LReLU
    for i, f in enumerate(list_f[1:]):
        name = "disc_conv2d_%s" % (i + 2)
        x = Convolution2D(f, 3, 3, subsample=(2, 2), name=name, border_mode="same", bias=False, init=conv2D_init,W_regularizer=l2(wd))(x)
        x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
        x = LeakyReLU(0.2)(x)
    #auxiliar classifier features
    #feats = GlobalAveragePooling2D()(x)
    #feats = Convolution2D(n_classes, 3, 3, name="last_conv", border_mode="same", bias=False, init=conv2D_init,W_regularizer=l2(wd))(x)
    #feats = GlobalAveragePooling2D()(feats)
  
    # Last convolution
    x = Convolution2D(1, 3, 3, name="aux_conv", border_mode="same", bias=False, init=conv2D_init,W_regularizer=l2(wd))(x)
    # Average pooling, it serves as traditional GAN single number true/fake
    x = GlobalAveragePooling2D()(x)

    if monsterClass: #2*nClasses (nClasses True, nClasses False) and no true/fake output
        aux = Dense(n_classes*2, activation='softmax', name='auxiliary')(aux_feats)
        discriminator_model = Model(input=[disc_input], output=aux, name=model_name)
    else:
        #FC7 = Dense(128, name='FC7',W_regularizer=l2(wd))(feats)
        #FC7 = LeakyReLU(0.2)(FC7)
        aux = Dense(n_classes, activation='softmax', name='auxiliary')(x)
        discriminator_model = Model(input=[disc_input], output=[x,aux], name=model_name)

    visualize_model(discriminator_model)

    return discriminator_model


def DCGAN(generator, discriminator, noise_dim, img_source_dim, img_dest_dim,monsterClass):
    """DCGAN generator + discriminator model

    Args:
        generator: keras generator model
        discriminator: keras discriminator model
        noise_dim: generator input noise dimension
        img_dim: real image data dimension

    Returns:
        keras model
    """
    noise_input = Input(shape=noise_dim, name="noise_input")
    image_input = Input(shape=img_source_dim, name="image_input")

    generated_image = generator([noise_input,image_input])
    if monsterClass:
        y_aux = discriminator(generated_image)
        DCGAN = Model(input=[noise_input,image_input],
                  output=y_aux,
                  name="DCGAN")
    else:
        DCGAN_output,y_aux = discriminator(generated_image)
        DCGAN = Model(input=[noise_input,image_input],
                  output=[DCGAN_output,y_aux],
                  name="DCGAN")
    visualize_model(DCGAN)

    return DCGAN
