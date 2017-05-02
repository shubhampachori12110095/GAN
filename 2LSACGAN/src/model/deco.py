import os
import sys
import time
import numpy as np
import models_WGAN as models
from keras.utils import generic_utils
# Utils
sys.path.append("../utils")
import general_utils
import data_utils
from data_utils import invNorm
import matplotlib.pyplot as plt
from IPython import display

def evaluating_GENned(noise_scale,noise_dim,X_source_test,Y_source_test,classifier,gen_model):
    #converting source test set into source-GENned set(passing through the GEN)
    n = data_utils.sample_noise(noise_scale, X_source_test.shape[0], noise_dim)
    X_gen_test = gen_model.predict ([n,X_source_test], batch_size=1024, verbose=0)
    loss4, acc4 = classifier.evaluate(X_gen_test, Y_source_test,batch_size=256, verbose=0)    
    print('\n Classifier Accuracy on source-GENned domain:  %.0f%%' % (100 * acc4))


def trainDECO(**kwargs):
    """
    Train standard DCGAN model

    args: **kwargs (dict) keyword arguments that specify the model hyperparameters
    """
    # Roll out the parameters
    generator = kwargs["generator"]
    discriminator = kwargs["discriminator"]
    dset = kwargs["dset"]
    img_dim = kwargs["img_dim"]
    nb_epoch = kwargs["nb_epoch"]
    batch_size = kwargs["batch_size"]
    n_batch_per_epoch = kwargs["n_batch_per_epoch"]
    bn_mode = kwargs["bn_mode"]
    noise_dim = kwargs["noise_dim"]
    noise_scale = kwargs["noise_scale"]
    lr_D = kwargs["lr_D"]
    lr_G = kwargs["lr_G"]
    opt_D = kwargs["opt_D"]
    opt_G = kwargs["opt_G"]
    clamp_lower = kwargs["clamp_lower"]
    clamp_upper = kwargs["clamp_upper"]
    image_dim_ordering = kwargs["image_dim_ordering"]
    epoch_size = n_batch_per_epoch * batch_size
    deterministic = kwargs["deterministic"]
    inject_noise = kwargs["inject_noise"]
    model = kwargs["model"]
    no_supertrain = kwargs["no_supertrain"]
    noClass = kwargs["noClass"]
    resume = kwargs["resume"]
    name = kwargs["name"]
    wd = kwargs["wd"]
    monsterClass = kwargs["monsterClass"]
    pretrained = kwargs["pretrained"]
    print("\nExperiment parameters:")
    for key in kwargs.keys():
        print key, kwargs[key]
    print("\n")

    # Setup environment (logging directory etc)
    general_utils.setup_logging("DCGAN")

    # Load and normalize data
    if dset == "mnistM":
        X_source_train,Y_source_train, _, _, n_classes1 = data_utils.load_image_dataset(img_dim, image_dim_ordering,dset='mnist')
        X_dest_train,Y_dest_train, _, _,n_classes2 = data_utils.load_image_dataset(img_dim, image_dim_ordering,dset='mnistM')
    elif dset == "washington_vandal50k":
        X_source_train = data_utils.load_image_dataset(img_dim, image_dim_ordering,dset='washington')
        X_dest_train = data_utils.load_image_dataset(img_dim, image_dim_ordering,dset='vandal50k')
    elif dset == "washington_vandal12classes":
        X_source_train = data_utils.load_image_dataset(img_dim, image_dim_ordering,dset='washington12classes')
        X_dest_train = data_utils.load_image_dataset(img_dim, image_dim_ordering,dset='vandal12classes')
    elif dset == "washington_vandal12classesNoBackground":
        X_source_train,Y_source_train,n_classes1 = data_utils.load_image_dataset(img_dim, image_dim_ordering,dset='washington12classes')
        X_dest_train,Y_dest_train,n_classes2 = data_utils.load_image_dataset(img_dim, image_dim_ordering,dset='vandal12classesNoBackground')
    elif dset == "Wash_Vand_12class_LMDB":
        X_source_train,Y_source_train,n_classes1 = data_utils.load_image_dataset(img_dim, image_dim_ordering,dset='Wash_12class_LMDB')
        X_dest_train,Y_dest_train,n_classes2 = data_utils.load_image_dataset(img_dim, image_dim_ordering,dset='Vand_12class_LMDB')
    elif dset == "Vand_Vand_12class_LMDB":
        X_source_train,Y_source_train,X_source_test, Y_source_test,n_classes1 = data_utils.load_image_dataset(img_dim, image_dim_ordering,dset='Vand_12class_LMDB_Background')
        X_dest_train,Y_dest_train, X_dest_test, Y_dest_test, n_classes2 = data_utils.load_image_dataset(img_dim, image_dim_ordering,dset='Vand_12class_LMDB')
    elif dset == "Wash_Color_LMDB":
        X_source_train,Y_source_train,X_source_test, Y_source_test,n_classes1 = data_utils.load_image_dataset(img_dim, image_dim_ordering,dset='Wash_Color_LMDB')
        X_dest_train,Y_dest_train, X_dest_test, Y_dest_test, n_classes2 = data_utils.load_image_dataset(img_dim, image_dim_ordering,dset='Wash_Color_LMDB')
    else:
        print "dataset not supported"
    if n_classes1 != n_classes2: #sanity check
        print "number of classes mismatch between source and dest domains"
    n_classes = n_classes1 

    # Get the full real image dimension
    img_source_dim = X_source_train.shape[-3:] # is it backend agnostic?
    img_dest_dim = X_dest_train.shape[-3:] 

    # Create optimizers
    opt_G = data_utils.get_optimizer(opt_G, lr_G)
    opt_D = data_utils.get_optimizer(opt_D, lr_D)
    opt_C = data_utils.get_optimizer('SGD', 0.01)

    #######################
    # Load models
    #######################
    noise_dim = (noise_dim,)
    generator_model = models.generator_upsampling_mnistM(noise_dim, img_source_dim,img_dest_dim, bn_mode,deterministic,inject_noise,wd, dset=dset)
    classifier = models.resnet(img_dest_dim,n_classes,wd=0.0001) #it is img_dest_dim because it is actually the generated image dim,that is equal to dest_dim
    GenToClassifierModel = models.GenToClassifierModel(generator_model, classifier, noise_dim, img_source_dim)

    #################
    # Load weight 
    ################
    if pretrained:
        model_path = "../../models/DCGAN"
        class_weights_path = os.path.join(model_path, 'NoBackground_100epochs.h5')
        classifier.load_weights(class_weights_path)
 #   if resume: ########loading previous saved model weights
 #       data_utils.load_model_weights(generator_model, discriminator_model, DCGAN_model, name)

    #######################
    # Compile models
    #######################
    generator_model.compile(loss='mse', optimizer=opt_G)
#    classifier.trainable = True # I wanna freeze the classifier without any training updates
    classifier.compile(loss='categorical_crossentropy', optimizer=opt_C,metrics=['accuracy']) # it is actually never using optimizer

    models.make_trainable(classifier, False)
    GenToClassifierModel.compile(loss='categorical_crossentropy', optimizer=opt_G,metrics=['accuracy'])

    #######################
    # Train classifier
    #######################
    if not pretrained: 
        loss1,acc1 =classifier.evaluate(X_dest_test, Y_dest_test,batch_size=256, verbose=0)
        print('\n Classifier Accuracy on target domain test set before training: %.00f%%' % (100.0 * acc1))
        classifier.fit(X_dest_train, Y_dest_train, validation_split=0.1, batch_size=512, nb_epoch=20, verbose=1)
    else:
        print ("Loaded pretrained classifier, computing accuracy on target domain test set:")
    loss2,acc2 = classifier.evaluate(X_dest_test, Y_dest_test,batch_size=512, verbose=0)
    print('\n Classifier Accuracy on target domain test set after training:  %.00f%%' % (100.0 * acc2))
    loss3, acc3 = classifier.evaluate(X_source_test, Y_source_test,batch_size=512, verbose=0)
    print('\n Classifier Accuracy on source domain test set:  %.00f%%' % (100.0 * acc3))
    evaluating_GENned(noise_scale,noise_dim,X_source_test,Y_source_test,classifier,generator_model)

#    model_path = "../../models/DCGAN"
#    class_weights_path = os.path.join(model_path, 'NoBackground_100epochs.h5')
#    classifier.save_weights(class_weights_path, overwrite=True)

#    models.make_trainable(classifier, False)
    #classifier.trainable = False # I wanna freeze the classifier without any more training updates
#    classifier.compile(loss='categorical_crossentropy', optimizer=opt_C,metrics=['accuracy']) 
    #################
    #  DECO training
    ################
    gen_iterations = 0
    for e in range(nb_epoch):
        # Initialize progbar and batch counter
        progbar = generic_utils.Progbar(epoch_size)
        batch_counter = 1
        start = time.time()
            ###################################
            # 1) Train the critic / discriminator
            ###################################
        list_class_loss_real = []
        X_dest_batch, Y_dest_batch,idx_dest_batch = next(data_utils.gen_batch(X_dest_train, Y_dest_train, batch_size))
        X_source_batch, Y_source_batch,idx_source_batch = next(data_utils.gen_batch(X_source_train, Y_source_train, batch_size))
        #######################
        # 2) Train the generator
        #######################
        X_gen = data_utils.sample_noise(noise_scale, batch_size*n_batch_per_epoch, noise_dim)
        X_source_batch2, Y_source_batch2,idx_source_batch2 = next(data_utils.gen_batch(X_source_train, Y_source_train, batch_size*n_batch_per_epoch))
        GenToClassifierModel.fit([X_gen, X_source_batch2], Y_source_batch2, batch_size=256, nb_epoch=1, verbose=1)

       
        gen_iterations += 1
        batch_counter += 1

            # plot images 1 times per epoch
        X_dest_batch_plot,Y_dest_batch_plot,idx_dest_plot = next(data_utils.gen_batch(X_dest_train,Y_dest_train, batch_size=32))
        X_source_batch_plot,Y_source_batch_plot,idx_source_plot = next(data_utils.gen_batch(X_source_train,Y_source_train, batch_size=32))

        data_utils.plot_generated_batch(X_dest_train,X_source_train, generator_model,
                                                 noise_dim, image_dim_ordering,idx_source_plot,batch_size=32)
        print ("Dest labels:") 
        print (Y_dest_train[idx_source_plot].argmax(1))
        print ("Source labels:") 
        print (Y_source_batch_plot.argmax(1))
        print('\nEpoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))

        # Save model weights (by default, every 5 epochs)
        #data_utils.save_model_weights(generator_model, discriminator_model, DCGAN_model, e, name)
        evaluating_GENned(noise_scale,noise_dim,X_source_test,Y_source_test,classifier,generator_model)

        loss3, acc3 = classifier.evaluate(X_source_test, Y_source_test,batch_size=512, verbose=0)
        print('\n Classifier Accuracy on source domain test set:  %.00f%%' % (100.0 * acc3))

