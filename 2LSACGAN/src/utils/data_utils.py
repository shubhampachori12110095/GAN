import os
import h5py
import numpy as np
from scipy import stats
from keras.utils import np_utils
from keras.datasets import mnist, cifar10
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import matplotlib
from keras.optimizers import Adam, SGD, RMSprop
import caffe
import lmdb
import PIL.Image
from StringIO import StringIO
import scipy.misc
from scipy import ndimage
from weightnorm import AdamWithWeightnorm, SGDWithWeightnorm
from keras import backend as K

def lr_decay(models,decay_value):
    for model in models:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr*decay_value)
        print("lr changed to {}".format(K.get_value(model.optimizer.lr)))

#from weightnorm import AdamWithWeightnorm
def sample_noise(noise_scale, batch_size, noise_dim):
    return np.random.normal(scale=noise_scale, size=(batch_size, noise_dim[0]))

def sample_noise_uniform(noise_scale, batch_size, noise_dim):
    return np.random.uniform(low=-noise_scale,high=noise_scale, size=(batch_size, noise_dim[0]))

def img_buffer(current, buffer,use, op):
#current is the current batch, buffer is the buffered previous batches, use is the numpy to be used in the discriminator
    if op == 'push':
        buffer[0:8] = np.copy(buffer[0:8] )
        buffer[8:16] = np.copy( buffer[24:32])  
        buffer[16:32] = np.copy(current[0:16])
        use = buffer
    if op == 'pop':
        use = np.copy(current)
        use[0:16] = np.copy(buffer[8:24]) 
    if op == 'init':
        buffer = np.copy(current)
        use=buffer
    return current, buffer, use

def normalization(X, image_dim_ordering):

    X = X / 255.
    if image_dim_ordering == "tf":
        X = (X - 0.5) / 0.5
    else:
        X = (X - 0.5) / 0.5
    return X


def inverse_normalization(X):
    return ((X * 0.5 + 0.5) * 255.).astype(np.uint8)

def invNorm(X):
    return X #((X * 0.5) * 255.).astype(np.uint8)

def read_lmdb(lmdb_file):
    cursor = lmdb.open(lmdb_file, readonly=True).begin().cursor()
    datum = caffe.proto.caffe_pb2.Datum()
    for _, value in cursor:
        datum.ParseFromString(value)
        s = StringIO()
        s.write(datum.data)
        s.seek(0)
        yield np.array(PIL.Image.open(s)), datum.label

def load_lmdb_datasets(image_dim_ordering, lmdb_dir, size=64,n_channels=1, max=50000):
    X=np.zeros((max,size,size,n_channels)) #tf dim order becaouse scipy imresize works with this
    Y=np.zeros((max,1))
    i=0
    for im, label in read_lmdb(lmdb_dir):
        im=scipy.misc.imresize(im, (size, size))
        if n_channels == 3: #it already has dimension (size,size,3)
            X[i]=im
        else:
            X[i]=np.expand_dims(im,4) #adding axis, so (size,size) --> (size,size,1)
        Y[i]=label
        i=i+1
        if i==max:
    	    break
    X = X.astype('float32')
    if image_dim_ordering == 'th':
        X=np.moveaxis(X, -1, 1) #switching from current tensorflow dim ordering to theano dim ordering
    X = normalization(X, image_dim_ordering)
    nb_classes = len(np.unique(Y))
    Y_ = np_utils.to_categorical(Y, nb_classes)

    if n_channels == 1:# TURNING 1 CHANNEL DATASET INTO 3-CHANNELS
        np.concatenate([X,X,X],axis=1)
    return X, Y_, nb_classes

def load_grayscale_npy(image_dim_ordering,path):
    #if os.path.exists(\"/home/torch/keras/KerasGAN/mnist-M_X_train.npy\"):
    X_train=np.load(path)
    if image_dim_ordering == 'th':
        X_train=np.expand_dims(X_train,1) # needed to get (nsamples,1,64,64) dimension
    else:
        X_train=np.expand_dims(X_train,4) # different image format:(nsamples,64,64,1) 
    X_train = X_train.astype('float32')
    X_train = normalization(X_train, image_dim_ordering)
    return X_train, Y_train, n_classes

def load_mnistM(image_dim_ordering):

    #Loading mnist labels, they should be the same of mnistM
    (_, y_train), (_, y_test) = mnist.load_data()
    X_train=np.load("../data/mnist-M_X_train.npy")
    X_test=np.load("../data/mnist-M_X_test.npy") 
    if image_dim_ordering == 'th':
        X_train = X_train.reshape(X_train.shape[0], 3, 28, 28)
        X_test = X_test.reshape(X_test.shape[0], 3, 28, 28)
    else:
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 3)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 3)
    #import code
    #code.interact(local=locals())
    #X_train = np.moveaxis(X_train,2,-1) #TEMPORARY SOLUTION
    #X_test = np.moveaxis(X_train,2,-1) #TEMPORARY SOLUTION
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = normalization(X_train, image_dim_ordering)
    X_test = normalization(X_test, image_dim_ordering)

    nb_classes = len(np.unique(np.hstack((y_train, y_test))))

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test,nb_classes


def load_mnist(image_dim_ordering):

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if image_dim_ordering == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
        X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    else:
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = normalization(X_train, image_dim_ordering)
    X_test = normalization(X_test, image_dim_ordering)

    nb_classes = len(np.unique(np.hstack((y_train, y_test))))

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, Y_train, X_test, Y_test, nb_classes


def load_cifar10(image_dim_ordering):

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    if image_dim_ordering == 'th':
        X_train = X_train.reshape(X_train.shape[0], 3, 32, 32)
        X_test = X_test.reshape(X_test.shape[0], 3, 32, 32)
    else:
        X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
        X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = normalization(X_train, image_dim_ordering)
    X_test = normalization(X_test, image_dim_ordering)

    nb_classes = len(np.unique(np.vstack((y_train, y_test))))

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, Y_train, X_test, Y_test, nb_classes


def load_celebA(img_dim, image_dim_ordering):

    with h5py.File("../../data/processed/CelebA_%s_data.h5" % img_dim, "r") as hf:

        X_dest_train = hf["data"][:].astype(np.float32)
        X_dest_train = normalization(X_dest_train, image_dim_ordering)

        if image_dim_ordering == "tf":
            X_dest_train = X_dest_train.transpose(0, 2, 3, 1)

        return X_dest_train


def load_image_dataset(img_dim, image_dim_ordering,dset='mnist'):
    #if dset == "celebA":
    #    X_train = load_celebA(img_dim, image_dim_ordering)
    if dset == "mnist":
        X_train, Y_train, X_test, Y_test, n_classes = load_mnist(image_dim_ordering)
    elif dset == "mnistM":
        X_train, Y_train, X_test, Y_test, n_classes = load_mnistM(image_dim_ordering)
    elif dset == "OfficeDslr":
        X_train, Y_train, n_classes = load_lmdb_datasets(image_dim_ordering,lmdb_dir='/home/paolo/SSD_backup/SSD/digits/digits/jobs/20170412-163620-984b/train_db/',n_channels=3,size=64,max=399)
        X_test, Y_test, n_classes = load_lmdb_datasets(image_dim_ordering,lmdb_dir=  '/home/paolo/SSD_backup/SSD/digits/digits/jobs/20170412-163620-984b/val_db/',n_channels=3,size=64,max=99)
    elif dset == "OfficeAmazon":
        X_train, Y_train, n_classes = load_lmdb_datasets(image_dim_ordering,lmdb_dir='/home/paolo/SSD_backup/SSD/digits/digits/jobs/20170412-163452-e0d2/train_db/',n_channels=3,size=64,max=2254)
        X_test, Y_test, n_classes = load_lmdb_datasets(image_dim_ordering,lmdb_dir=  '/home/paolo/SSD_backup/SSD/digits/digits/jobs/20170412-163452-e0d2/val_db/',n_channels=3,size=64,max=563)
    elif dset == "OfficeWebcam":
        X_train, Y_train, n_classes = load_lmdb_datasets(image_dim_ordering,lmdb_dir='/home/paolo/SSD_backup/SSD/digits/digits/jobs/20170412-163555-1c57/train_db/',n_channels=3,size=64,max=637)
        X_test, Y_test, n_classes = load_lmdb_datasets(image_dim_ordering,lmdb_dir=  '/home/paolo/SSD_backup/SSD/digits/digits/jobs/20170412-163555-1c57/val_db/',n_channels=3,size=64,max=158)
    elif dset == "Vand_12class_LMDB":
        X_train, Y_train, n_classes = load_lmdb_datasets(image_dim_ordering,lmdb_dir='/home/paolo/SSD_backup/SSD/digits/digits/jobs/20170324-162456-9bc1/train_db/',size=64,max=20000)
        X_test, Y_test, n_classes = load_lmdb_datasets(image_dim_ordering,lmdb_dir='/home/paolo/SSD_backup/SSD/digits/digits/jobs/20170324-162456-9bc1/val_db/',size=64,max=10000)
    elif dset == "Wash_12class_LMDB":
        X_train, Y_train, n_classes = load_lmdb_datasets(image_dim_ordering,lmdb_dir='/home/paolo/SSD_backup/SSD/digits/digits/jobs/20170324-112522-707b/train_db/',size=64,max=10000)
        X_test, Y_test, n_classes = load_lmdb_datasets(image_dim_ordering,lmdb_dir='/home/paolo/SSD_backup/SSD/digits/digits/jobs/20170324-112522-707b/val_db/',size=64,max=10000)
    elif dset == "Vand_12class_LMDB_Background":
        X_train, Y_train, n_classes = load_lmdb_datasets(image_dim_ordering,lmdb_dir='/home/paolo/SSD_backup/SSD/digits/digits/jobs/20170324-113729-28f3/train_db/',size=64,max=10000)
        X_test, Y_test, n_classes = load_lmdb_datasets(image_dim_ordering,lmdb_dir='/home/paolo/SSD_backup/SSD/digits/digits/jobs/20170324-113729-28f3/val_db/',size=64,max=10000)
    elif dset == "bedrooms128":
        X_train, Y_train, n_classes = load_lmdb_datasets(image_dim_ordering,lmdb_dir='/home/paolo/SSD_backup/SSD/digits/digits/jobs/20170427-192135-3259/train_db/',n_channels=3,size=128,max=32000)
        X_test, Y_test, n_classes = load_lmdb_datasets(image_dim_ordering,lmdb_dir=  '/home/paolo/SSD_backup/SSD/digits/digits/jobs/20170427-192135-3259/val_db/',n_channels=3,size=128,max=320)
    elif dset == "bedrooms128small":
        X_train, Y_train, n_classes = load_lmdb_datasets(image_dim_ordering,lmdb_dir='/home/paolo/SSD_backup/SSD/digits/digits/jobs/20170427-192135-3259/train_db/',n_channels=3,size=128,max=32)
        X_test, Y_test, n_classes = load_lmdb_datasets(image_dim_ordering,lmdb_dir=  '/home/paolo/SSD_backup/SSD/digits/digits/jobs/20170427-192135-3259/val_db/',n_channels=3,size=128,max=32)
    elif dset == "bedrooms":
        X_train, Y_train, n_classes = load_lmdb_datasets(image_dim_ordering,lmdb_dir='/home/paolo/SSD_backup/SSD/digits/digits/jobs/20170419-131209-267a/train_db/',n_channels=3,size=64,max=100000)
        X_test, Y_test, n_classes = load_lmdb_datasets(image_dim_ordering,lmdb_dir=  '/home/paolo/SSD_backup/SSD/digits/digits/jobs/20170419-131209-267a/val_db/',n_channels=3,size=64,max=320)
    elif dset == "bedrooms_small":
        X_train, Y_train, n_classes = load_lmdb_datasets(image_dim_ordering,lmdb_dir='/home/paolo/SSD_backup/SSD/digits/digits/jobs/20170419-131209-267a/train_db/',n_channels=3,size=64,max=128)
        X_test, Y_test, n_classes = load_lmdb_datasets(image_dim_ordering,lmdb_dir=  '/home/paolo/SSD_backup/SSD/digits/digits/jobs/20170419-131209-267a/val_db/',n_channels=3,size=64,max=128)
    elif dset == "Wash_Color_LMDB":
        X_train, Y_train, n_classes = load_lmdb_datasets(image_dim_ordering,lmdb_dir='/home/paolo/SSD_backup/SSD/digits/digits/jobs/20161108-142358-d9b8/train_db/',
                                                         n_channels=3, size=64, max=10000)
        X_test, Y_test, n_classes = load_lmdb_datasets(image_dim_ordering,lmdb_dir='/home/paolo/SSD_backup/SSD/digits/digits/jobs/20161108-142358-d9b8/val_db/',
                                                         n_channels=3, size=64, max=10000)


    return X_train, Y_train, X_test, Y_test, n_classes

    #if dset == "cifar10":
    #    X_train, _, _, _ = load_cifar10(image_dim_ordering)
    #if dset == "vandal50k":
    #    X_train = load_grayscale_npy(image_dim_ordering,path='../data/vandal_50ksamples64x64.npy')
    #if dset == "washington":
    #    X_train = load_grayscale_npy(image_dim_ordering,path='../data/Washington_Split0_train64x64.npy')
    #if dset == "vandal12classes":
    #    X_train = load_grayscale_npy(image_dim_ordering,path='../data/VANDAL_12classes_train.npy')
    #if dset == "vandal12classesNoBackground":
    #    X_train, Y_train, n_classes = load_grayscale_npy(image_dim_ordering,path='../data/VANDAL_12classesNoBackground.npy')
    #if dset == "washington12classes":
    #    X_train = load_grayscale_npy(image_dim_ordering,path='../data/Washington_normalized_12classes_train.npy')


def load_toy(n_mixture=8, std=0.01, radius=1.0, pts_per_mixture=5000):

    thetas = np.linspace(0, 2 * np.pi, n_mixture + 1)[:-1]
    xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
    cov = std * np.eye(2)

    X = np.zeros((n_mixture * pts_per_mixture, 2))

    for i in range(n_mixture):

        mean = np.array([xs[i], ys[i]])
        pts = np.random.multivariate_normal(mean, cov, pts_per_mixture)
        X[i * pts_per_mixture: (i + 1) * pts_per_mixture, :] = pts

    return X


def get_optimizer(opt, lr):

    if opt == "SGD":
        return SGD(lr=lr)
    elif opt == "RMSprop":
        return RMSprop(lr=lr)
    elif opt == "Adam":
        return Adam(lr=lr, beta_1=0.5)
    elif opt == "AdamWithWeightnorm":
        return AdamWithWeightnorm(lr=lr, beta_1=0.5)
    elif opt == "SGDWithWeightnorm":
        return SGDWithWeightnorm(lr=lr)



def gen_batch(X, Y, batch_size):
    while True:
        idx = np.random.choice(X.shape[0], batch_size, replace=False)
        yield X[idx], Y[idx], idx



def get_disc_batch(X_dest_batch, generator_model, batch_counter, batch_size, noise_dim,X_source_batch, noise_scale=0.5):

    X_disc_real = X_dest_batch[:batch_size]
    source_images = X_source_batch[:batch_size]
    # Pass noise to the generator
    noise_input = sample_noise(noise_scale, batch_size, noise_dim)

    # Produce an output
    X_disc_gen = generator_model.predict([noise_input,source_images])

    return X_disc_real, X_disc_gen


def save_model_weights(generator_model, discriminator_model, DCGAN_model, e, name,classifier_model=None,zclass_model=None):

    model_path = "../../models/DCGAN"

    if e % 5 == 0:
        gen_weights_path = os.path.join(model_path, name+'_gen.h5')
        generator_model.save_weights(gen_weights_path, overwrite=True)

        disc_weights_path = os.path.join(model_path, name+'_disc.h5')
        discriminator_model.save_weights(disc_weights_path, overwrite=True)

        DCGAN_weights_path = os.path.join(model_path, name+'_DCGAN.h5')
        DCGAN_model.save_weights(DCGAN_weights_path, overwrite=True)
        if classifier_model is not None:
            class_weights_path = os.path.join(model_path, name+'_class.h5')
            classifier_model.save_weights(class_weights_path, overwrite=True)
        if zclass_model is not None:
            zclass_weights_path = os.path.join(model_path, name+'_zclass.h5')
            zclass_model.save_weights(zclass_weights_path, overwrite=True)

def load_model_weights(generator_model, discriminator_model, DCGAN_model, name, classifier_model=None):
    model_path = "../../models/DCGAN"

    gen_weights_path = os.path.join(model_path, name+'_gen.h5')
    generator_model.load_weights(gen_weights_path)

    disc_weights_path = os.path.join(model_path, name+'_disc.h5')
    discriminator_model.load_weights(disc_weights_path)

    DCGAN_weights_path = os.path.join(model_path, name+'_DCGAN.h5')
    DCGAN_model.load_weights(DCGAN_weights_path)

    if classifier_model is not None:       
        class_weights_path = os.path.join(model_path, name+'_class.h5')
        classifier_model.load_weights(class_weights_path)

def plot_generated_batch(X_dest,X_source, generator_model, noise_dim, image_dim_ordering, idx, noise_scale=0.5,batch_size=32):
    plt.figure(figsize=(20,20))
    # Generate images
#    X_gen = sample_noise_uniform(noise_scale, batch_size, noise_dim)
    X_gen = sample_noise(noise_scale, batch_size, noise_dim)
    source_images = X_source[idx]
    dest_images = X_dest[idx]
#    source_images = X_source[np.random.randint(0,X_source.shape[0],size=batch_size),:,:,:]
    X_gen = generator_model.predict([X_gen,source_images])
    
    dest_images = inverse_normalization(dest_images)
    X_gen = inverse_normalization(X_gen)
    source_images = inverse_normalization(source_images)
    Xg = X_gen[:32]
    Xr = dest_images[:32]
    Xs = source_images[:32]
    if image_dim_ordering == "tf":
        Axis=1
    if image_dim_ordering == "th":
        Axis=2

    list_rows_gen = []
    list_rows_source = []
    list_rows_dest = []
    for i in range(int(Xs.shape[0] / 8)):
        Xr1 = np.concatenate(    [Xr[k] for k in range(8 * i, 8 * (i + 1))], axis=Axis)
        Xsource = np.concatenate([Xs[k] for k in range(8 * i, 8 * (i + 1))], axis=Axis)
        Xg1 = np.concatenate(    [Xg[k] for k in range(8 * i, 8 * (i + 1))], axis=Axis)
        list_rows_dest.append(Xr1)
        list_rows_gen.append(Xg1)
        list_rows_source.append(Xsource)

    Xr = np.concatenate(list_rows_dest, axis=Axis-1)
    Xg = np.concatenate(list_rows_gen, axis=Axis-1)
    Xs = np.concatenate(list_rows_source, axis=Axis-1)
    Xr = Xr.transpose(1,2,0)
    Xs = Xs.transpose(1,2,0)
    Xg = Xg.transpose(1,2,0)
    plt.subplot(311)
    if Xr.shape[-1] == 1:
        plt.imshow(Xr[:, :, 0], cmap="gray")
    else:
        plt.imshow(Xr)
    plt.subplot(312)
    if Xg.shape[-1] == 1:
        plt.imshow(Xg[:, :, 0], cmap="gray")
    else:
        plt.imshow(Xg)
    plt.subplot(313)
    if Xs.shape[-1] == 1:
        plt.imshow(Xs[:, :, 0], cmap="gray")
    else:
        plt.imshow(Xs)
    plt.show()

    plt.pause(0.01)    
    #plt.savefig("../../figures/current_batch.png")
    #plt.clf()
    #plt.close()

def create_stepped_cols(n,m): # n = number of cols
    out = np.zeros((n,m,n))
    r = np.linspace(-0.5,0.5,m)
    d = np.arange(n)
    out[d,:,d] = r
    out.shape = (-1,n)
    np.maximum.accumulate(out, axis=0, out = out)
    return out

def slerp(val, low, high):
    """Spherical interpolation. val has a range of 0 to 1."""
    if val <= 0:
        return low
    elif val >= 1:
        return high
    elif np.allclose(low, high):
        return low
    omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high

#def lerp(val, low, high):
#    """Linear interpolation"""
#    return low + (high - low) * val

def BIG_ASS_VISUALIZATION_slerp(X_image, generator_model,noise_dim):
    X_noise = sample_noise(0.5, 10, noise_dim)
    count=0
    #X_image = ndimage.zoom(X_image, ( 10, 10))     
    for i in range(X_noise.shape[0] - 1):
        for j in range(100):
            #import code
            count = count + 1
            x = slerp(val=(j+1)/100.0,low=X_noise[i],high=X_noise[i+1])
            x = x[np.newaxis]
            X_gen = generator_model.predict([x,X_image[np.newaxis]])
            X_gen = inverse_normalization(X_gen)
            X_gen=X_gen[0,:,:,:] 
            X_gen = X_gen.transpose(1,2,0)
            filename = '/home/paolo/Downloads/video/image_%d.png' % count    
            im =PIL.Image.fromarray(X_gen)
            im = im.resize((140,140), PIL.Image.ANTIALIAS)
            im.save(filename)
    print 'finished'

def BIG_ASS_VISUALIZATION(X_image, generator_model):
    X_noise = create_stepped_cols(100,30)
    #import code
    #code.interact(local=locals())
    #X_image = ndimage.zoom(X_image, ( 10, 10))     
    for i in range(X_noise.shape[0]):
        X_gen = generator_model.predict([X_noise[i][np.newaxis],X_image[np.newaxis][np.newaxis]])
        X_gen = inverse_normalization(X_gen)
        X_gen=X_gen[0,:,:,:] 
        X_gen = X_gen.transpose(1,2,0)
        filename = '/home/paolo/Downloads/video/image_%d.png' % i    
        im =PIL.Image.fromarray(X_gen)
        im = im.resize((280,280), PIL.Image.ANTIALIAS)
        im.save(filename)
    print 'finished'

def BIG_ASS_VISUALIZATION2(X_image, generator_model):
    X_noise = np.zeros([1,100])

    #X_image = ndimage.zoom(X_image, ( 10, 10))     
    for i in range(3000):
        #import code
        #code.interact(local=locals())
        rand = sample_noise_gaussian(0.005, 1, (100,))
        X_noise = X_noise + rand
        np.clip(X_noise,-0.5,0.5,X_noise)
        X_gen = generator_model.predict([X_noise,X_image[np.newaxis][np.newaxis]])
        X_gen = inverse_normalization(X_gen)
        X_gen=X_gen[0,:,:,:] 
        X_gen = X_gen.transpose(1,2,0)
        filename = '/home/paolo/Downloads/video/image_%d.png' % i    
        im =PIL.Image.fromarray(X_gen)
        im = im.resize((280,280), PIL.Image.ANTIALIAS)
        im.save(filename)
    print 'finished'

def plot_debug(X_dest,X_source, generator_model, noise_dim, image_dim_ordering, idx, noise_scale=0.5,batch_size=32):
    plt.figure(figsize=(20,20))
    # Generate images
#    X_gen = sample_noise_uniform(noise_scale, batch_size, noise_dim)
    X_gen = sample_noise(noise_scale, batch_size, noise_dim)
    source_images = X_source[:32]
    dest_images = X_dest[:32]
#    source_images = X_source[np.random.randint(0,X_source.shape[0],size=batch_size),:,:,:]
    X_gen = generator_model.predict([X_gen,source_images])
    
    dest_images = inverse_normalization(dest_images)
    X_gen = inverse_normalization(X_gen)
    source_images = inverse_normalization(source_images)

    Xg = X_gen[:32]
    Xr = dest_images[:32]
    Xs = source_images[:32]
    if image_dim_ordering == "tf":
        Axis=1
    if image_dim_ordering == "th":
        Axis=2

    list_rows_gen = []
    list_rows_source = []
    list_rows_dest = []
    for i in range(int(Xs.shape[0] / 8)):
        Xr1 = np.concatenate(    [Xr[k] for k in range(8 * i, 8 * (i + 1))], axis=Axis)
        Xsource = np.concatenate([Xs[k] for k in range(8 * i, 8 * (i + 1))], axis=Axis)
        Xg1 = np.concatenate(    [Xg[k] for k in range(8 * i, 8 * (i + 1))], axis=Axis)
        list_rows_dest.append(Xr1)
        list_rows_gen.append(Xg1)
        list_rows_source.append(Xsource)

    Xr = np.concatenate(list_rows_dest, axis=Axis-1)
    Xg = np.concatenate(list_rows_gen, axis=Axis-1)
    Xs = np.concatenate(list_rows_source, axis=Axis-1)
    Xr = Xr.transpose(1,2,0)
    Xs = Xs.transpose(1,2,0)
    Xg = Xg.transpose(1,2,0)
    plt.subplot(311)
    if Xr.shape[-1] == 1:
        plt.imshow(Xr[:, :, 0], cmap="gray")
    else:
        plt.imshow(Xr)
    plt.subplot(312)
#    plt.show()
    if Xg.shape[-1] == 1:
        plt.imshow(Xg[:, :, 0], cmap="gray")
    else:
        plt.imshow(Xg)
#    plt.figure(figsize=(14,14))
#    plt.show()
    plt.subplot(313)
    if Xs.shape[-1] == 1:
        plt.imshow(Xs[:, :, 0], cmap="gray")
    else:
        plt.imshow(Xs)
#    plt.figure(figsize=(14,14))
    plt.show()

    plt.pause(0.01)    
    #plt.savefig("../../figures/current_batch.png")
    #plt.clf()
    #plt.close()


def plot_generated_toy_batch(X_dest, generator_model, discriminator_model, noise_dim, gen_iter, noise_scale=0.5):

    # Generate images
    X_gen = sample_noise(noise_scale, 10000, noise_dim)
    X_gen = generator_model.predict(X_gen)

    # Get some toy data to plot KDE of real data
    data = load_toy(pts_per_mixture=200)
    x = data[:, 0]
    y = data[:, 1]
    xmin, xmax = -1.5, 1.5
    ymin, ymax = -1.5, 1.5

    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    # Plot the contour
    fig = plt.figure(figsize=(10,10))
    plt.suptitle("Generator iteration %s" % gen_iter, fontweight="bold", fontsize=22)
    ax = fig.gca()
    ax.contourf(xx, yy, f, cmap='Blues', vmin=np.percentile(f,80), vmax=np.max(f), levels=np.linspace(0.25, 0.85, 30))

    # Also plot the contour of the discriminator
    delta = 0.025
    xmin, xmax = -1.5, 1.5
    ymin, ymax = -1.5, 1.5
    # Create mesh
    XX, YY = np.meshgrid(np.arange(xmin, xmax, delta), np.arange(ymin, ymax, delta))
    arr_pos = np.vstack((np.ravel(XX), np.ravel(YY))).T
    # Get Z = predictions
    ZZ = discriminator_model.predict(arr_pos)
    ZZ = ZZ.reshape(XX.shape)
    # Plot contour
    ax.contour(XX, YY, ZZ, cmap="Blues", levels=np.linspace(0.25, 0.85, 10))
    dy, dx = np.gradient(ZZ)
    # Add streamlines
    # plt.streamplot(XX, YY, dx, dy, linewidth=0.5, cmap="magma", density=1, arrowsize=1)
    # Scatter generated data
    plt.scatter(X_gen[:1000, 0], X_gen[:1000, 1], s=20, color="coral", marker="o")

    l_gen = plt.Line2D((0,1),(0,0), color='coral', marker='o', linestyle='', markersize=20)
    l_D = plt.Line2D((0,1),(0,0), color='steelblue', linewidth=3)
    l_real = plt.Rectangle((0, 0), 1, 1, fc="steelblue")

    # Create legend from custom artist/label lists
    # bbox_to_anchor = (0.4, 1)
    ax.legend([l_real, l_D, l_gen], ['Real data KDE', 'Discriminator contour',
                                     'Generated data'], fontsize=18, loc="upper left")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax + 0.8)
    plt.show()
    plt.pause(0.01)
    #plt.savefig("../../figures/toy_dataset_iter%s.jpg" % gen_iter)
    #plt.clf()
    #plt.close()


if __name__ == '__main__':

    data = load_toy(pts_per_mixture=200)

    x = data[:, 0]
    y = data[:, 1]
    xmin, xmax = -1.5, 1.5
    ymin, ymax = -1.5, 1.5

    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    fig = plt.figure()
    gen_it = 5
    plt.suptitle("Generator iteration %s" % gen_it, fontweight="bold")
    ax = fig.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # Contourf plot
    cfset = ax.contourf(xx, yy, f, cmap='Blues', vmin=np.percentile(f,90),
                        vmax=np.max(f), levels=np.linspace(0.25, 0.85, 30))
    # cfset = ax.contour(xx, yy, f, color="k", levels=np.linspace(0.25, 0.85, 30), label="roger")
    plt.legend()
    plt.show()
