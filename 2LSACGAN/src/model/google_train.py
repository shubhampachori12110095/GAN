# -*- coding: utf-8 -*-
import os
import sys
import time
import numpy as np
import google_models as models
from keras.utils import generic_utils
# Utils
sys.path.append("../utils")
import general_utils
import data_utils
from data_utils import *
from image_history_buffer import *
import matplotlib.pyplot as plt
from IPython import display
import code
from google_models import *
from additional_models import *
from weightnorm import *
from collections import deque

def train(**kwargs):
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
    use_mbd = kwargs["use_mbd"]
    image_dim_ordering = kwargs["image_dim_ordering"]
    epoch_size = n_batch_per_epoch * batch_size
    deterministic = kwargs["deterministic"]
    inject_noise = kwargs["inject_noise"]
    model = kwargs["model"]
    no_supertrain = kwargs["no_supertrain"]
    pureGAN = kwargs["pureGAN"]
    lsmooth = kwargs["lsmooth"]
    disc_type = kwargs["disc_type"]
    resume = kwargs["resume"]
    name = kwargs["name"]
    wd = kwargs["wd"]
    history_size = kwargs["history_size"]
    monsterClass = kwargs["monsterClass"]
    print("\nExperiment parameters:")
    for key in kwargs.keys():
        print key, kwargs[key]
    print("\n")

    # Setup environment (logging directory etc)
    general_utils.setup_logging("DCGAN")

    # Load and normalize data
    if dset == "mnistM":
        X_source_train,Y_source_train, X_source_test, Y_source_test, n_classes1 = data_utils.load_image_dataset(img_dim, image_dim_ordering,dset='mnist',shuff=True)
#        X_source_train=np.concatenate([X_source_train,X_source_train,X_source_train], axis=1)
#        X_source_test=np.concatenate([X_source_test,X_source_test,X_source_test], axis=1)
        X_dest_train,Y_dest_train, X_dest_test, Y_dest_test, n_classes2 = data_utils.load_image_dataset(img_dim, image_dim_ordering,dset='mnistM',shuff=True)
    elif dset == "OfficeDslrToAmazon":
        X_source_train,Y_source_train,X_source_test, Y_source_test,n_classes1 = data_utils.load_image_dataset(img_dim, image_dim_ordering,dset='OfficeDslr')
        X_dest_train,Y_dest_train,X_dest_test, Y_dest_test, n_classes2 = data_utils.load_image_dataset(img_dim, image_dim_ordering,dset='OfficeAmazon')
    else:
        print "dataset not supported"
    if n_classes1 != n_classes2: #sanity check
        print "number of classes mismatch between source and dest domains"
    n_classes = n_classes1 #


    img_source_dim = X_source_train.shape[-3:] # is it backend agnostic?
    img_dest_dim = X_dest_train.shape[-3:] 

    # Create optimizers
    opt_D = data_utils.get_optimizer(opt_D, lr_D)
    opt_G = data_utils.get_optimizer(opt_G, lr_G)
    opt_GC = data_utils.get_optimizer('Adam', lr_G / 10.0)
    opt_C = data_utils.get_optimizer('Adam', lr_D)
    opt_Z = data_utils.get_optimizer('Adam', lr_G)

    #######################
    # Load models
    #######################
    noise_dim = (noise_dim,)
    generator_model = models.generator_google_mnistM(noise_dim, img_source_dim,img_dest_dim,deterministic,pureGAN,wd)
#    discriminator_model = models.discriminator_google_mnistM(img_dest_dim, wd)       
    discriminator_model = models.discriminator_dcgan(img_dest_dim, wd,n_classes,disc_type)       
    classificator_model = models.classificator_google_mnistM(img_dest_dim,n_classes, wd)
    DCGAN_model = models.DCGAN_naive(generator_model, discriminator_model, noise_dim, img_source_dim)
    zclass_model = z_coerence(generator_model,img_source_dim, bn_mode,wd,inject_noise,n_classes,noise_dim, model_name="zClass")
#    GenToClassifier_model = models.GenToClassifierModel(generator_model, classificator_model, noise_dim, img_source_dim)
    #disc_penalty_model = models.disc_penalty(discriminator_model,noise_dim,img_source_dim,opt_D,model_name="disc_penalty_model")    
    zclass_model = z_coerence(generator_model,img_source_dim, bn_mode,wd,inject_noise,n_classes,noise_dim, model_name="zClass")

    ############################
    # Compile models
    ############################
    generator_model.compile(loss='mse', optimizer=opt_G)

    models.make_trainable(discriminator_model, False)
    models.make_trainable(classificator_model, False)
#    models.make_trainable(disc_penalty_model, False)
    if model == 'wgan':
        DCGAN_model.compile(loss=models.wasserstein, optimizer=opt_G)
        models.make_trainable(discriminator_model, True)
   #     models.make_trainable(disc_penalty_model, True)
        discriminator_model.compile(loss=models.wasserstein, optimizer=opt_D)
    if model == 'lsgan':
        if disc_type == "simple_disc":
            DCGAN_model.compile(loss=['mse'], optimizer=opt_G)
            models.make_trainable(discriminator_model, True)
            discriminator_model.compile(loss=['mse'], optimizer=opt_D)
        elif disc_type == "nclass_disc":
            DCGAN_model.compile(loss=['mse','categorical_crossentropy'],loss_weights=[1.0, 0.1], optimizer=opt_G) 
            models.make_trainable(discriminator_model, True)
            discriminator_model.compile(loss=['mse','categorical_crossentropy'], loss_weights=[1.0, 0.1], optimizer=opt_D)
#    GenToClassifier_model.compile(loss='categorical_crossentropy', optimizer=opt_GC)
    models.make_trainable(classificator_model, True)
    classificator_model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=opt_C)
    zclass_model.compile(loss=['mse'],optimizer = opt_Z)

    visualize = False
    ########        
    #MAKING TRAIN+TEST numpy array for global testing:
    ########
    Xtarget_dataset = np.concatenate([X_dest_train,X_dest_test],axis=0)    
    Ytarget_dataset = np.concatenate([Y_dest_train,Y_dest_test],axis=0)    

    if resume: ########loading previous saved model weights and checking actual performance
 #       data_utils.load_model_weights(generator_model, discriminator_model, DCGAN_model, name)
        data_utils.load_model_weights(generator_model, discriminator_model, DCGAN_model, name,classificator_model)
        loss4, acc4 = classificator_model.evaluate(Xtarget_dataset, Ytarget_dataset,batch_size=1024, verbose=0)                                                                                   
        print('\n Classifier Accuracy on full target domain:  %.2f%%' % (100 * acc4))

    else:
        X_gen = data_utils.sample_noise(noise_scale, X_source_train.shape[0], noise_dim)
        zclass_loss = zclass_model.fit([X_gen,X_source_train],[X_gen],batch_size=256,epochs=10)
    ####train zclass regression model only if not resuming:

    gen_iterations = 0
    max_history_size = int( history_size * batch_size) 
    img_buffer = ImageHistoryBuffer((0,)+img_source_dim, max_history_size, batch_size, n_classes)
    #################
    # Start training
    ################
    for e in range(nb_epoch):
        # Initialize progbar and batch counter
        progbar = generic_utils.Progbar(epoch_size)
        batch_counter = 1
        start = time.time()

        while batch_counter < n_batch_per_epoch:
            if no_supertrain is None:
                if (gen_iterations < 25) and (not resume):
                    disc_iterations = 100
                if gen_iterations % 500 == 0:
                    disc_iterations = 100
                else:
                    disc_iterations = kwargs["disc_iterations"]
            else:
                if (gen_iterations <25) and (not resume):
                    disc_iterations = 100
                else:
                    disc_iterations = kwargs["disc_iterations"]

            ###################################
            # 1) Train the critic / discriminator
            ###################################
            list_disc_loss_real = deque(10*[0],10)
            list_disc_loss_gen = deque(10*[0],10)
            list_gen_loss = deque(10*[0],10)
            list_zclass_loss = deque(10*[0],10)
            list_classifier_loss = deque(10*[0],10)
            list_gp_loss = deque(10*[0],10)
            for disc_it in range(disc_iterations):
                X_dest_batch, Y_dest_batch,idx_dest_batch = next(data_utils.gen_batch(X_dest_train, Y_dest_train, batch_size))
                X_source_batch, Y_source_batch,idx_source_batch = next(data_utils.gen_batch(X_source_train, Y_source_train, batch_size))
                ##########
                # Create a batch to feed the discriminator model
                #########
                X_disc_real, X_disc_gen = data_utils.get_disc_batch(X_dest_batch, generator_model, batch_counter, batch_size, noise_dim, X_source_batch, noise_scale=noise_scale)

                # Update the discriminator
                if model == 'wgan':
                    current_labels_real = -np.ones(X_disc_real.shape[0]) 
                    current_labels_gen = np.ones(X_disc_gen.shape[0]) 
                elif model == 'lsgan': 
                    if disc_type == "simple_disc":
                        current_labels_real = np.ones(X_disc_real.shape[0]) 
                        current_labels_gen = np.zeros(X_disc_gen.shape[0]) 
                    elif disc_type == "nclass_disc":
                        virtual_real_labels =np.zeros([X_disc_gen.shape[0],n_classes])
                        current_labels_real = [np.ones(X_disc_real.shape[0]),virtual_real_labels]
                        current_labels_gen =[np.zeros(X_disc_gen.shape[0]), Y_source_batch ]
                ##############
                #Train the disc on gen-buffered samples and on current real samples
                ##############
                disc_loss_real = discriminator_model.train_on_batch(X_disc_real, current_labels_real)
                img_buffer.add_to_buffer(X_disc_gen,current_labels_gen, batch_size)
                bufferImages, bufferLabels = img_buffer.get_from_buffer(batch_size)
                disc_loss_gen = discriminator_model.train_on_batch(bufferImages, bufferLabels)

                #if not isinstance(disc_loss_real, collections.Iterable): disc_loss_real = [disc_loss_real]
                #if not isinstance(disc_loss_real, collections.Iterable): disc_loss_gen = [disc_loss_gen]
                if disc_type == "simple_disc":
                    list_disc_loss_real.appendleft(disc_loss_real)
                    list_disc_loss_gen.appendleft(disc_loss_gen)
                elif disc_type == "nclass_disc":
                    list_disc_loss_real.appendleft(disc_loss_real[0])
                    list_disc_loss_gen.appendleft(disc_loss_gen[0])
                #############
                ####Train the discriminator w.r.t gradient penalty
                #############
                #gp_loss = disc_penalty_model.train_on_batch([X_disc_real,X_disc_gen],current_labels_real) #dummy labels,not used in the loss function
                #list_gp_loss.appendleft(gp_loss)

            ################
            ###CLASSIFIER TRAINING OUTSIDE DISC LOOP(wanna train in just 1 time even if disc_iter > 1)
            #################
            class_loss_gen = classificator_model.train_on_batch(X_disc_gen, Y_source_batch*0.7) #LABEL SMOOTHING!!!!
            list_classifier_loss.appendleft(class_loss_gen[1])
            #######################
            # 2) Train the generator
            #######################
            X_gen = data_utils.sample_noise(noise_scale, batch_size, noise_dim)
            X_source_batch2, Y_source_batch2,idx_source_batch2 = next(data_utils.gen_batch(X_source_train, Y_source_train, batch_size))
            if model == 'wgan':
                gen_loss = DCGAN_model.train_on_batch([X_gen,X_source_batch2], -np.ones(X_gen.shape[0]))
            if model == 'lsgan':
                if disc_type == "simple_disc":                
                    gen_loss =  DCGAN_model.train_on_batch([X_gen,X_source_batch2], np.ones(X_gen.shape[0])) #TRYING SAME BATCH OF DISC
                elif disc_type == "nclass_disc":
                    gen_loss = DCGAN_model.train_on_batch([X_gen,X_source_batch2], [np.ones(X_gen.shape[0]),Y_source_batch2])
                    gen_loss = gen_loss[0]
            list_gen_loss.appendleft(gen_loss)


            zclass_loss = zclass_model.train_on_batch([X_gen,X_source_batch2],[X_gen])
            list_zclass_loss.appendleft(zclass_loss)
            ##############
            #Train the generator w.r.t the aux classifier:
            #############
#            GenToClassifier_model.train_on_batch([X_gen,X_source_batch2],Y_source_batch2)


           # I SHOULD TRY TO CLASSIFY EVEN ON DISCRIMINATOR, PUTTING ONE CLASS FOR REAL SAMPLES AND N CLASS FOR FAKE

            gen_iterations += 1
            batch_counter += 1

            progbar.add(batch_size, values=[("Loss_D_real", np.mean(list_disc_loss_real)),
                                            ("Loss_D_gen", np.mean(list_disc_loss_gen)),
                                            ("Loss_G", np.mean(list_gen_loss)),
                                            ("Loss_Z", np.mean(list_zclass_loss)),
                                            ("Loss_Classifier",np.mean(list_classifier_loss))])

            # plot images 1 times per epoch
            if batch_counter % (n_batch_per_epoch) == 0:
                X_source_batch_plot,Y_source_batch_plot,idx_source_plot = next(data_utils.gen_batch(X_source_test,Y_source_test, batch_size=32))
                data_utils.plot_generated_batch(X_dest_test,X_source_test, generator_model,noise_dim, image_dim_ordering,idx_source_plot,batch_size=32)
            if gen_iterations % (n_batch_per_epoch*5) == 0:
                if visualize:
                    BIG_ASS_VISUALIZATION_slerp(X_source_train[1], generator_model, noise_dim)
        #    if (e % 20) == 0:
        #        lr_decay([discriminator_model,DCGAN_model,classificator_model],decay_value=0.95)

        print ("Dest labels:") 
        print (Y_dest_test[idx_source_plot].argmax(1))
        print ("Source labels:") 
        print (Y_source_batch_plot.argmax(1))
        print('\nEpoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))

        # Save model weights (by default, every 5 epochs)
        data_utils.save_model_weights(generator_model, discriminator_model, DCGAN_model, e, name,classificator_model,zclass_model)

        #testing accuracy of trained classifier 
        loss4, acc4 = classificator_model.evaluate(Xtarget_dataset, Ytarget_dataset,batch_size=1024, verbose=0)                                                                                   
        print('\n Classifier Accuracy and loss on full target domain:  %.2f%% / %.5f%%' % ((100 * acc4), loss4))
        #decay learning rates by multiplying each model optimizer.lr by decay_value:

#        loss2,acc2 = classifier.evaluate(X_dest_test, Y_dest_test,batch_size=512, verbose=0)                                                                                         
#        print('\n Classifier Accuracy on target domain test set after training:  %.2f%%' % (100 * acc2))                                                                             
#        loss3, acc3 = classifier.evaluate(X_source_test, Y_source_test,batch_size=512, verbose=0)                                                                                    
#        print('\n Classifier Accuracy on source domain test set:  %.2f%%' % (100 * acc3)) 
#        evaluating_GENned(noise_scale,noise_dim,X_source_test,Y_source_test,classifier,generator_model)


