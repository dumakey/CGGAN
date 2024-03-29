# -*- coding: utf-8 -*-
import os
import re
from shutil import rmtree, copytree
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
import pickle
import cv2 as cv
from random import randint

import tensorflow as tf

import reader
import dataset_processing
import models
import dataset_augmentation
import postprocessing


class CGGAN:

    def __init__(self, launch_file):

        self.parameters = self.create_container()
        self.datasets = self.create_container()
        self.model = self.create_container()
        self.predictions = self.create_container()

        # Setup general parameters
        casedata = reader.read_case_setup(launch_file)
        self.parameters.analysis = casedata.analysis
        self.parameters.training_parameters = casedata.training_parameters
        self.parameters.img_processing = casedata.img_processing
        self.parameters.img_size = casedata.img_resize
        self.parameters.samples_generation = casedata.samples_generation
        self.parameters.data_augmentation = casedata.data_augmentation
        self.parameters.activation_plotting = casedata.activation_plotting
        self.case_dir = casedata.case_dir

        # Sensitivity analysis variable identification
        sens_vars = [item for item in self.parameters.training_parameters.items() if
                     item[0] not in ('enc_hidden_layers', 'dec_hidden_layers') if type(item[1]) == list]
        self.parameters.sens_variable = sens_vars[0] if len(sens_vars) != 0 else None

        # Check for model reconstruction
        if self.parameters.analysis['import'] == True:
            self.model.imported = True
            generator, discriminator = self.reconstruct_model()
            self.model.Model = [self.create_container()]
            self.model.Model[0].Generator = generator
            self.model.Model[0].Discriminator = discriminator
        else:
            self.model.imported = False

    def __str__(self):
        class_name = type(self).__name__

        return '{}, a class to generate contours based on Generative Adversarial Neural Networks (GANNs)'.format(class_name)

    def create_container(self):

        class container:
            pass

        return container
        
    def launch_analysis(self):

        analysis_ID = self.parameters.analysis['type']
        analysis_list = {
                        'singletraining': self.singletraining,
                        'sensanalysis': self.sensitivity_analysis_on_training,
                        'traingenerate': self.traingenerate,
                        'generate': self.contour_generation,
                        'datagen': self.data_generation,
                        'plotactivations': self.plot_activations,
                        }

        analysis_list[analysis_ID]()

    def sensitivity_analysis_on_training(self):

        # Retrieve sensitivity variable
        sens_variable = self.parameters.sens_variable

        case_dir = self.case_dir
        training_size = self.parameters.training_parameters['train_size']
        batch_size = self.parameters.training_parameters['batch_size']
        img_size = self.parameters.img_size

        self.datasets.data_train, self.datasets.data_cv, self.datasets.data_test = \
        dataset_processing.get_datasets(case_dir,training_size,img_size)

        self.datasets.dataset_train, self.datasets.dataset_cv, self.datasets.dataset_test = \
        dataset_processing.get_tensorflow_datasets(self.datasets.data_train,self.datasets.data_cv,self.datasets.data_test,batch_size)
        
        if self.model.imported == False:
            self.train_model(sens_variable)
        self.export_model_performance(sens_variable)
        self.export_model(sens_variable)
        self.export_nn_log()

    def singletraining(self):

        case_dir = self.case_dir
        training_size = self.parameters.training_parameters['train_size']
        batch_size = self.parameters.training_parameters['batch_size']
        img_size = self.parameters.img_size

        self.datasets.data_train, self.datasets.data_cv, self.datasets.data_test = \
        dataset_processing.get_datasets(case_dir,training_size,img_size)

        dataset_train, dataset_cv, dataset_test = dataset_processing.get_tensorflow_datasets(self.datasets.data_train,self.datasets.data_cv,
                                                                                self.datasets.data_test,batch_size)
        self.datasets.dataset_train = dataset_train
        self.datasets.dataset_cv = dataset_cv
        self.datasets.dataset_test = dataset_test

        if self.model.imported == False:
            self.train_model()
        self.export_model_performance()
        self.export_model()
        self.export_nn_log()
    
    def traingenerate(self):
    
        # Training
        case_dir = self.case_dir
        training_size = self.parameters.training_parameters['train_size']
        batch_size = self.parameters.training_parameters['batch_size']
        img_size = self.parameters.img_size

        self.datasets.data_train, self.datasets.data_cv, self.datasets.data_test = \
        dataset_processing.get_datasets(case_dir,training_size,img_size)

        self.datasets.dataset_train, self.datasets.dataset_cv, self.datasets.dataset_test = \
        dataset_processing.get_tensorflow_datasets(self.datasets.data_train,self.datasets.data_cv,self.datasets.data_test,batch_size)

        if self.model.imported == False:
            self.train_model()
        self.export_model_performance()
        self.export_model()
        self.export_nn_log()
        
        # Generation
        model_dir = os.path.join(case_dir,'Results',str(self.parameters.analysis['case_ID']),'Model')
        generation_dir = os.path.join(case_dir,'Results','pretrained_model')
        if os.path.exists(generation_dir):
            rmtree(generation_dir)
        copytree(model_dir,generation_dir)
        self.model.imported = True
        self.contour_generation()
        

    def data_generation(self):

        transformations = [{k:v[1:] for (k,v) in self.parameters.img_processing.items() if v[0] == 1}][0]
        augdata_size = self.parameters.data_augmentation[1]
        self.generate_augmented_data(transformations,augdata_size)

    def plot_activations(self):

        # Parameters
        case_dir = self.case_dir
        img_size = self.parameters.img_size
        batch_size = self.parameters.training_parameters['batch_size']
        training_size = self.parameters.training_parameters['train_size']
        n = self.parameters.activation_plotting['n_samples']
        figs_per_row = self.parameters.activation_plotting['n_cols']
        rows_to_cols_ratio = self.parameters.activation_plotting['rows2cols_ratio']

        # Generate datasets
        self.datasets.data_train, self.datasets.data_cv, self.datasets.data_test = \
        dataset_processing.get_datasets(case_dir,training_size,img_size)

        self.datasets.dataset_train, self.datasets.dataset_cv, self.datasets.dataset_test = \
        dataset_processing.get_tensorflow_datasets(self.datasets.data_train,self.datasets.data_cv,self.datasets.data_test,batch_size)

        m_tr = self.datasets.data_train[0].shape[0]
        m_cv = self.datasets.data_cv[0].shape[0]
        m_ts = self.datasets.data_test[0].shape[0]
        m = m_tr + m_cv + m_ts

        # Read datasets
        dataset = np.zeros((m,*img_size),dtype='uint8')
        dataset[:m_tr,:] = self.datasets.data_train[0]
        dataset[m_tr:m_tr+m_cv,:] = self.datasets.data_cv[0]
        dataset[m_tr+m_cv:m,:] = self.datasets.data_test[0]

        # Index image sampling
        idx = [randint(0,m-1) for i in range(n)]
        idx_set = set(idx)
        while len(idx) != len(idx_set):
            extra_item = randint(1,m)
            idx_set.add(extra_item)

        # Reconstruct encoder model
        _, Discriminator = self.reconstruct_model()

        # Plot
        for idx in idx_set:
            img = dataset[idx,:]
            postprocessing.monitor_hidden_layers(img,Discriminator,case_dir,figs_per_row,rows_to_cols_ratio,idx)

    def generate_augmented_data(self, transformations, augmented_dataset_size=1):

        # Set storage folder for augmented dataset
        augmented_dataset_dir = os.path.join(self.case_dir,'Datasets','Augmented')

        # Unpack data
        X = dataset_processing.read_dataset(case_folder=self.case_dir,dataset_folder='To_augment')
        # Generate new dataset
        data_augmenter = dataset_augmentation.datasetAugmentationClass(X,transformations,augmented_dataset_size,augmented_dataset_dir)
        data_augmenter.transform_images()
        data_augmenter.export_augmented_dataset()

    def contour_generation(self):

        if self.model.imported == True:
            storage_dir = os.path.join(self.case_dir,'Results','pretrained_model','Image_generation')
        else:
            storage_dir = os.path.join(self.case_dir,'Results','Image_generation')
        if os.path.exists(storage_dir):
            rmtree(storage_dir)
        os.makedirs(storage_dir)

        # Read parameters
        case_dir = self.case_dir
        casedata = reader.read_case_logfile(os.path.join(case_dir,'Results','pretrained_model','CGGAN.log'))
        img_size = casedata.img_size

        if self.model.imported == False:
            self.singletraining()

        ## GENERATE NEW DATA - SAMPLING ##
        X_samples = self.generate_samples(casedata)
        postprocessing.plot_generated_samples(X_samples,img_size,storage_dir)

    def train_model(self, sens_var=None):

        # Parameters
        image_shape = (self.parameters.img_size[1],self.parameters.img_size[0],1)  # (height, width, channels)
        nepoch = self.parameters.training_parameters['epochs']
        epoch_iter = self.parameters.training_parameters['epoch_iter']
        num_iter = nepoch * epoch_iter
        batch_size = self.parameters.training_parameters['batch_size']
        batch_shape = (batch_size,*image_shape)
        sample_shape = (1,*image_shape)
        noise_dim = self.parameters.training_parameters['noise_dim']
        alpha = self.parameters.training_parameters['learning_rate']
        l1_reg = self.parameters.training_parameters['l1_reg']
        l2_reg = self.parameters.training_parameters['l2_reg']
        l3_reg = self.parameters.training_parameters['l3_reg']
        dropout = self.parameters.training_parameters['dropout']
        activation = self.parameters.training_parameters['activation']

        # Create model containers
        if sens_var != None:
            # compute the sweep number
            if type(alpha) == list:
                N = len(alpha)
            elif type(l1_reg) == list:
                N = len(l1_reg)
            elif type(l2_reg) == list:
                N = len(l2_reg)
            elif type(l3_reg) == list:
                N = len(l3_reg)
            elif type(dropout) == list:
                N = len(dropout)
            elif type(activation) == list:
                N = len(activation)            
            elif type(noise_dim) == list:
                N = len(noise_dim)
        else:
            N = 1

        # List conversion
        alpha = [alpha if type(alpha) != list else alpha[i] for i in range(N)]
        noise_dim = [noise_dim if type(noise_dim) != list else noise_dim[i] for i in range(N)]
        l1_reg = [l1_reg if type(l1_reg) != list else l1_reg[i] for i in range(N)]
        l2_reg = [l2_reg if type(l2_reg) != list else l2_reg[i] for i in range(N)]
        l3_reg = [l3_reg if type(l3_reg) != list else l3_reg[i] for i in range(N)]
        activation = [activation if type(activation) != list else activation[i] for i in range(N)]
        dropout = [dropout if type(dropout) != list else dropout[i] for i in range(N)]

        self.model.Model = [self.create_container() for i in range(N)]
        self.model.History = [self.create_container() for i in range(N)]
        self.model.Optimizers = [self.create_container() for i in range(N)]

        for i in range(N):
            # Training variables
            self.model.History[i].disc_loss_train = np.zeros([nepoch,])
            self.model.History[i].disc_metric_train = np.zeros([nepoch,])
            self.model.History[i].gen_loss_train = np.zeros([nepoch,])
            self.model.History[i].gen_metric_train = np.zeros([nepoch,])
            # Validation variables
            self.model.History[i].disc_loss_cv = np.zeros([nepoch,])
            self.model.History[i].disc_metric_cv = np.zeros([nepoch,])
            self.model.History[i].gen_loss_cv = np.zeros([nepoch,])
            self.model.History[i].gen_metric_cv = np.zeros([nepoch,])

            # Models and functions declaration
            discriminator = self.model.Model[i].Discriminator = models.Discriminator(activation[i],l2_reg[i],l1_reg[i],dropout[i])
            generator = self.model.Model[i].Generator = models.Generator(image_shape[:2],activation[i],l2_reg[i],l1_reg[i],dropout[i])
            disc_optimizer = self.model.Optimizers[i].disc_optimizer = models.optimizer(2*alpha[i])
            gen_optimizer = self.model.Optimizers[i].gen_optimizer = models.optimizer(alpha[i])
            loss = models.loss_function
            metric_disc = models.performance_metric
            metric_gen = models.performance_metric

            epoch = 1
            disc_streaming_loss = 0
            disc_streaming_loss_cv = 0
            disc_streaming_metric = 0
            disc_streaming_metric_cv = 0
            gen_streaming_loss = 0
            gen_streaming_loss_cv = 0
            gen_streaming_metric = 0
            gen_streaming_metric_cv = 0

            # Create iterator
            train_iterator = iter(self.datasets.dataset_train)
            for j in range(1,num_iter+1):
                ### Update discriminator
                real_image_batch = tf.reshape(train_iterator.get_next(),batch_shape)
                noise_batch = tf.random.normal((batch_size,noise_dim[i]))
                fake_image_batch = generator(noise_batch)
                # Ground truth labels
                fake_label_batch = tf.zeros((batch_size,1)) + 0.05 * tf.random.uniform((batch_size,1))
                real_label_batch = tf.ones((batch_size,1)) + 0.05 * tf.random.uniform((batch_size,1))
                with tf.GradientTape() as tape_disc:
                    # Prediction computations
                    fake_logit_batch = discriminator(fake_image_batch)
                    real_logit_batch = discriminator(real_image_batch)
                    # Loss computation
                    fake_loss_batch = loss('disc',discriminator,fake_logit_batch,fake_label_batch,real_logit_batch,
                                           real_label_batch,fake_image_batch,real_image_batch,l3_reg[i])
                    disc_loss_batch = fake_loss_batch + sum(discriminator.losses)
                    # Metric computation
                    fake_metric_batch = metric_disc(fake_logit_batch,fake_label_batch).numpy()
                    real_metric_batch = metric_disc(real_logit_batch,real_label_batch).numpy()
                    disc_metric_batch = 0.5 * (fake_metric_batch + real_metric_batch)
                # Weights update
                disc_gradients = tape_disc.gradient(disc_loss_batch,discriminator.trainable_weights)
                disc_optimizer.apply_gradients(zip(disc_gradients,discriminator.trainable_weights))

                # Misleading labels
                misleading_label_batch = tf.ones((batch_size,1))

                ### Update generator
                noise_batch = tf.random.normal((batch_size,noise_dim[i]))
                with tf.GradientTape() as tape_gen:
                    fake_image_batch = generator(noise_batch)
                    fake_logit_batch = discriminator(fake_image_batch)
                    gen_loss_batch = loss('gen',None,fake_logit_batch,misleading_label_batch,None,None,None,None,0.0) + sum(generator.losses)
                    gen_metric_batch = metric_gen(fake_logit_batch,misleading_label_batch).numpy()
                gen_gradients = tape_gen.gradient(gen_loss_batch,generator.trainable_weights)
                gen_optimizer.apply_gradients(zip(gen_gradients,generator.trainable_weights))

                disc_streaming_loss += disc_loss_batch
                disc_streaming_metric += disc_metric_batch
                gen_streaming_loss += gen_loss_batch
                gen_streaming_metric += gen_metric_batch

                if j % epoch_iter == 0:
                    # Cancel regularization terms for discriminator & generator
                    discriminator.set_up_CV_state()
                    generator.set_up_CV_state()

                    # Evaluate on training dataset
                    self.model.History[i].disc_loss_train[epoch-1] = disc_streaming_loss/epoch_iter
                    self.model.History[i].disc_metric_train[epoch-1] = disc_streaming_metric/epoch_iter
                    self.model.History[i].gen_loss_train[epoch-1] = gen_streaming_loss/epoch_iter
                    self.model.History[i].gen_metric_train[epoch-1] = gen_streaming_metric/epoch_iter
                    # Evaluate on cross-validation dataset
                    niter = 0
                    cv_iterator = iter(self.datasets.dataset_cv) # create iterator for cross-validation dataset
                    while niter < 10:
                        ## Evaluate discriminator
                        real_image_cv = tf.reshape(cv_iterator.get_next(),sample_shape)
                        noise_cv = tf.random.normal((1,noise_dim[i]))
                        fake_image_cv = generator(noise_cv)
                        # Prediction computations
                        fake_logit_cv = discriminator(fake_image_cv)
                        real_logit_cv = discriminator(real_image_cv)
                        # Ground truth labels
                        fake_label_cv = tf.zeros((1,1))
                        real_label_cv = tf.ones((1,1))
                        # Loss computation
                        disc_loss_cv = loss('disc',discriminator,fake_logit_cv,fake_label_cv,real_logit_cv,real_label_cv,fake_image_cv,real_image_cv,0.0)
                        # Metric computation
                        fake_metric_cv = metric_disc(fake_logit_cv,fake_label_cv).numpy()
                        real_metric_cv = metric_disc(real_logit_cv,real_label_cv).numpy()
                        disc_metric_cv = 0.5 * (fake_metric_cv + real_metric_cv)

                        ## Evaluate generator
                        noise_cv = tf.random.normal((1,noise_dim[i]))
                        fake_image_cv = generator(noise_cv)
                        fake_logit_cv = discriminator(fake_image_cv)
                        misleading_label_cv = tf.ones((1,1))
                        gen_loss_cv = loss('gen',None,fake_logit_batch,misleading_label_cv,None,None,None,None,0.0)
                        gen_metric_cv = metric_disc(fake_logit_cv,misleading_label_cv).numpy()

                        disc_streaming_loss_cv += disc_loss_cv
                        disc_streaming_metric_cv += disc_metric_cv
                        gen_streaming_loss_cv += gen_loss_cv
                        gen_streaming_metric_cv += gen_metric_cv
                        niter += 1

                    self.model.History[i].disc_loss_cv[epoch-1] = disc_streaming_loss_cv/niter
                    self.model.History[i].disc_metric_cv[epoch-1] = disc_streaming_metric_cv/niter
                    self.model.History[i].gen_loss_cv[epoch-1] = gen_streaming_loss_cv/niter
                    self.model.History[i].gen_metric_cv[epoch-1] = gen_streaming_metric_cv/niter
                    niter = 0
                    discriminator.set_up_training_state()  # cancel regularization terms for discriminator
                    generator.set_up_training_state()  # cancel regularization terms for generator

                    # Print results
                    print('Epoch {}, Discriminator loss (T,CV): ({:.2f},{:.2f}), Discriminator metric (T,CV): ({:.3f},{:.3f}) || '
                          'Generator loss (T,CV): ({:.2f},{:.2f}), Generator metric (T,CV): ({:.3f},{:.3f})'
                          .format(epoch,
                                  self.model.History[i].disc_loss_train[epoch-1],
                                  self.model.History[i].disc_loss_cv[epoch-1],
                                  self.model.History[i].disc_metric_train[epoch-1],
                                  self.model.History[i].disc_metric_cv[epoch-1],
                                  self.model.History[i].gen_loss_train[epoch-1],
                                  self.model.History[i].gen_loss_cv[epoch-1],
                                  self.model.History[i].gen_metric_train[epoch-1],
                                  self.model.History[i].gen_metric_cv[epoch-1],))

                    # Reset streaming variables
                    disc_streaming_loss = 0
                    disc_streaming_loss_cv = 0
                    disc_streaming_metric = 0
                    disc_streaming_metric_cv = 0
                    gen_streaming_loss = 0
                    gen_streaming_loss_cv = 0
                    gen_streaming_metric = 0
                    gen_streaming_metric_cv = 0
                    epoch += 1

            print()

    def generate_samples(self, parameters):

        ## BUILD DECODER ##
        noise_dim = parameters.training_parameters['noise_dim']
        n_samples = self.parameters.samples_generation['n_samples']

        # Retrieve decoder weights
        X_samples = []
        for model in self.model.Model:
            ## SAMPLE IMAGES ##
            noise_batch = tf.random.normal((n_samples,noise_dim))
            samples = model.Generator(noise_batch)
            X_samples.append(samples)

        return X_samples

    def export_model_performance(self, sens_var=None):

        try:
            History = self.model.History
        except:
            raise Exception('There is no evolution data for this model. Train model first.')
        else:
            if type(History) == list:
                N = len(History)
            else:
                N = 1
                History = [History]

            # load / save checkpoint
            case_ID = self.parameters.analysis['case_ID']
            results_folder = os.path.join(self.case_dir,'Results',str(case_ID),'Model_performance')
            if os.path.exists(results_folder):
                rmtree(results_folder)
            os.makedirs(results_folder)

            nepoch = self.parameters.training_parameters['epochs']
            epochs = np.arange(1,nepoch+1,1)
            delimiter = ';'
            for j,h in enumerate(History):
                disc_loss_train = h.disc_loss_train
                disc_loss_cv = h.disc_loss_cv
                disc_metric_train = h.disc_metric_train
                disc_metric_cv = h.disc_metric_cv
                gen_loss_train = h.gen_loss_train
                gen_loss_cv = h.gen_loss_cv
                gen_metric_train = h.gen_metric_train
                gen_metric_cv = h.gen_metric_cv

                ## LOGS ##
                if sens_var != None:
                    if type(sens_var[1][j]) == str:
                        results_folder = os.path.join(self.case_dir,'Results',str(case_ID),'Model_performance','{}={}'
                                                      .format(sens_var[0],sens_var[1][j]))
                    else:
                        results_folder = os.path.join(self.case_dir,'Results',str(case_ID),'Model_performance','{}={:.3f}'
                                                      .format(sens_var[0],sens_var[1][j]))
                    os.mkdir(results_folder)
                
                # Export discriminator logs
                loss_filepath = os.path.join(results_folder,'CGGAN_discriminator_loss.csv')
                with open(loss_filepath,'w') as f:
                    f.write('Epoch{}Training{}CV\n'.format(delimiter,delimiter))
                    for i in range(nepoch):
                        f.write('%d%s%.2f%s%.2f\n' %(i+1,delimiter,disc_loss_train[i],delimiter,disc_loss_cv[i]))
                metrics_filepath = os.path.join(results_folder,'CGGAN_discriminator_metrics.csv')
                with open(metrics_filepath,'w') as f:
                    f.write('Epoch{}Training{}CV\n'.format(delimiter,delimiter))
                    for i in range(nepoch):
                        f.write('%d%s%.2f%s%.2f\n' %(i+1,delimiter,disc_metric_train[i],delimiter,disc_metric_cv[i]))
                # Export generator logs
                loss_filepath = os.path.join(results_folder,'CGGAN_generator_loss.csv')
                with open(loss_filepath,'w') as f:
                    f.write('Epoch{}Training{}CV\n'.format(delimiter,delimiter))
                    for i in range(nepoch):
                        f.write('%d%s%.2f%s%.2f\n' %(i+1,delimiter,gen_loss_train[i],delimiter,gen_loss_cv[i]))
                metrics_filepath = os.path.join(results_folder,'CGGAN_generator_metrics.csv')
                with open(metrics_filepath,'w') as f:
                    f.write('Epoch{}Training{}CV\n'.format(delimiter,delimiter))
                    for i in range(nepoch):
                        f.write('%d%s%.2f%s%.2f\n' %(i+1,delimiter,gen_metric_train[i],delimiter,gen_metric_cv[i]))

                ## PLOTS ##
                # Discriminator loss
                fig_disc, ax_disc = plt.subplots(2,1)
                ax_disc[0].plot(epochs,disc_loss_train,label='Training',color='r')
                ax_disc[0].plot(epochs,disc_loss_cv,label='Cross-validation',color='b')
                ax_disc[1].plot(epochs,disc_metric_train,label='Training',color='r')
                ax_disc[1].plot(epochs,disc_metric_cv,label='Cross-validation',color='b')
                ax_disc[0].grid()
                ax_disc[1].grid()
                ax_disc[1].set_xlabel('Epochs',size=12)
                ax_disc[0].set_ylabel('Loss',size=12)
                ax_disc[1].set_ylabel('Accuracy',size=12)
                ax_disc[0].tick_params('both',labelsize=10)
                ax_disc[1].tick_params('both',labelsize=10)
                ax_disc[0].legend()
                plt.suptitle('Discriminator loss/accuracy evolution case = {}'.format(str(case_ID)))

                # Generator loss
                fig_gen, ax_gen = plt.subplots(2,1)
                ax_gen[0].plot(epochs,gen_loss_train,label='Training',color='r')
                ax_gen[0].plot(epochs,gen_loss_cv,label='Cross-validation',color='b')
                ax_gen[1].plot(epochs,gen_metric_train,label='Training',color='r')
                ax_gen[1].plot(epochs,gen_metric_cv,label='Cross-validation',color='b')
                ax_gen[0].grid()
                ax_gen[1].grid()
                ax_gen[1].set_xlabel('Epochs',size=12)
                ax_gen[0].set_ylabel('Loss',size=12)
                ax_gen[1].set_ylabel('Accuracy',size=12)
                ax_gen[0].tick_params('both',labelsize=10)
                ax_gen[1].tick_params('both',labelsize=10)
                ax_gen[0].legend()
                plt.suptitle('Generator loss/accuracy evolution case = {}'.format(str(case_ID)))

                if sens_var:
                    disc_loss_plot_filename = 'Discriminator_performance_evolution_{}_{}={}.png'.format(str(case_ID),sens_var[0],str(sens_var[1][j]))
                    gen_loss_plot_filename = 'Generator_performance_evolution_{}_{}={}.png'.format(str(case_ID),sens_var[0],str(sens_var[1][j]))
                else:
                    disc_loss_plot_filename = 'Discriminator_performance_evolution_{}.png'.format(str(case_ID))
                    gen_loss_plot_filename = 'Generator_performance_evolution_{}.png'.format(str(case_ID))

                fig_disc.savefig(os.path.join(results_folder,disc_loss_plot_filename),dpi=200)
                fig_gen.savefig(os.path.join(results_folder,gen_loss_plot_filename),dpi=200)
                plt.close('all')

    def export_model(self, sens_var=None):

        N = len(self.model.Model)

        # Parameters
        case_ID = self.parameters.analysis['case_ID']
        alpha = self.parameters.training_parameters['learning_rate']
        noise_dim = self.parameters.training_parameters['noise_dim']
        img_dim = (self.parameters.img_size[1],self.parameters.img_size[0], 1)

        # List conversion
        alpha = [alpha if type(alpha) != list else alpha[i] for i in range(N)]
        noise_dim = [noise_dim if type(noise_dim) != list else noise_dim[i] for i in range(N)]

        for i in range(N):
            if sens_var:
                if type(sens_var[1][i]) == str:
                    storage_dir = os.path.join(self.case_dir,'Results',str(case_ID),'Model','{}={}'
                                               .format(sens_var[0],sens_var[1][i]))
                else:
                    storage_dir = os.path.join(self.case_dir,'Results',str(case_ID),'Model','{}={:.3f}'
                                               .format(sens_var[0],sens_var[1][i]))
                discriminator_model_name = 'CGGAN_discriminator_model_{}_{}={}'.format(str(case_ID),sens_var[0],str(sens_var[1][i]))
                generator_model_name = 'CGGAN_generator_model_{}_{}={}'.format(str(case_ID),sens_var[0],str(sens_var[1][i]))
            else:
                storage_dir = os.path.join(self.case_dir,'Results',str(case_ID),'Model')
                discriminator_model_name = 'CGGAN_discriminator_model_{}'.format(str(case_ID))
                generator_model_name = 'CGGAN_generator_model_{}'.format(str(case_ID))

            if os.path.exists(storage_dir):
                rmtree(storage_dir)
            os.makedirs(storage_dir)
            
            ### SAVE MODELS
            compilation_parameters = {'optimizer':models.optimizer(alpha[i]),'loss':models.cost_function(),
                                      'metric':models.base_metric()}
            # Save generator model
            # convert model to keras (to ease the process of loading and saving)
            generator_keras = models.convert_to_keras_model(self.model.Model[i].Generator,noise_dim[i],img_dim,compilation_parameters)
            generator_keras.save(os.path.join(storage_dir,generator_model_name))            
            
            # Save discriminator model
            # convert model to keras (to ease the process of loading and saving)
            discriminator_keras = models.convert_to_keras_model(self.model.Model[i].Discriminator,img_dim,1,compilation_parameters)
            discriminator_keras.save(os.path.join(storage_dir,discriminator_model_name))


    def reconstruct_model(self, mode='train'):

        storage_dir = os.path.join(self.case_dir,'Results','pretrained_model')
        try:
            casedata = reader.read_case_logfile(os.path.join(storage_dir,'CGGAN.log'))
            img_dim = (casedata.img_size[1],casedata.img_size[0],1)  # (height, width, channels)
            noise_dim = casedata.training_parameters['noise_dim']
            activation = casedata.training_parameters['activation']
            alpha = casedata.training_parameters['learning_rate']

            optimizer = models.optimizer(alpha)
            loss = models.cost_function()
            metric = models.base_metric()
            compilation_parameters = {'optimizer':optimizer,'loss':loss,'metric':metric}

            # Load Generator model
            generator = models.Generator(img_dim[:2],activation,0.0,0.0,0.0)
            generator_keras = models.convert_to_keras_model(generator,noise_dim,img_dim,compilation_parameters)

            generator_folder = [item for item in os.listdir(storage_dir) if os.path.isdir(os.path.join(storage_dir,item))
                                if item.startswith('CGGAN_generator_model')][0]
            generator_keras.load_weights(os.path.join(storage_dir,generator_folder))

            # Load Discriminator model
            discriminator = models.Discriminator(activation,0.0,0.0,0.0)
            discriminator_keras = models.convert_to_keras_model(discriminator,img_dim,1,compilation_parameters)

            discriminator_folder = [item for item in os.listdir(storage_dir) if os.path.isdir(os.path.join(storage_dir,item))
                                    if item.startswith('CGGAN_discriminator_model')][0]
            discriminator_keras.load_weights(os.path.join(storage_dir,discriminator_folder))


        except:
            print('There is no model stored in the folder')

        return generator_keras, discriminator_keras

    def export_nn_log(self):
    
        def update_log(parameters, model):
            training = OrderedDict()
            training['TRAINING SIZE'] = parameters.training_parameters['train_size']
            training['LEARNING RATE'] = parameters.training_parameters['learning_rate']
            training['L3 REGULARIZER'] = parameters.training_parameters['l3_reg']
            training['L2 REGULARIZER'] = parameters.training_parameters['l2_reg']
            training['L1 REGULARIZER'] = parameters.training_parameters['l1_reg']
            training['DROPOUT'] = parameters.training_parameters['dropout']
            training['ACTIVATION'] = parameters.training_parameters['activation']
            training['NUMBER OF EPOCHS'] = parameters.training_parameters['epochs']
            training['BATCH SIZE'] = parameters.training_parameters['batch_size']
            training['NOISE DIMENSION'] = parameters.training_parameters['noise_dim']
            training['DISCRIMINATOR OPTIMIZER'] = [optimizer.disc_optimizer._name for optimizer in model.Optimizers]
            training['GENERATOR OPTIMIZER'] = [optimizer.gen_optimizer._name for optimizer in model.Optimizers]

            analysis = OrderedDict()
            analysis['CASE ID'] = parameters.analysis['case_ID']
            analysis['ANALYSIS'] = parameters.analysis['type']
            analysis['IMPORTED MODEL'] = parameters.analysis['import']
            analysis['DISCRIMINATOR LAST TRAINING LOSS'] = ['{:.3f}'.format(history.disc_loss_train[-1]) for history in model.History]
            analysis['DISCRIMINATOR LAST CV LOSS'] = ['{:.3f}'.format(history.disc_loss_cv[-1]) for history in model.History]
            analysis['GENERATOR LAST TRAINING LOSS'] = ['{:.3f}'.format(history.gen_loss_train[-1]) for history in model.History]
            analysis['GENERATOR LAST CV LOSS'] = ['{:.3f}'.format(history.gen_loss_cv[-1]) for history in model.History]

            architecture = OrderedDict()
            architecture['INPUT SHAPE'] = parameters.img_size

            return training, analysis, architecture

        if self.parameters.analysis['type'] == 'sensanalysis':
            varname, varvalues = self.parameters.sens_variable
            for value in varvalues:
                self.parameters.training_parameters[varname] = value
                training, analysis, architecture = update_log(self.parameters,self.model)

                case_ID = self.parameters.analysis['case_ID']
                if type(value) == str:
                    storage_folder = os.path.join(self.case_dir,'Results',str(case_ID),'Model','{}={}'.format(varname,value))
                else:
                    storage_folder = os.path.join(self.case_dir,'Results',str(case_ID),'Model','{}={:.3f}'.format(varname,value))
                with open(os.path.join(storage_folder,'CGGAN.log'),'w') as f:
                    f.write('CGGAN log file\n')
                    f.write('==================================================================================================\n')
                    f.write('->ANALYSIS\n')
                    for item in analysis.items():
                        f.write(item[0] + '=' + str(item[1]) + '\n')
                    f.write('SENSITIVITY VARIABLE='+varname+'\n')
                    f.write('SENSITIVITY VALUES='+str(varvalues)+'\n')
                    f.write('--------------------------------------------------------------------------------------------------\n')
                    f.write('->TRAINING\n')
                    for item in training.items():
                        f.write(item[0] + '=' + str(item[1]) + '\n')
                    f.write('--------------------------------------------------------------------------------------------------\n')
                    f.write('->ARCHITECTURE\n')
                    for item in architecture.items():
                        f.write(item[0] + '=' + str(item[1]) + '\n')
                    f.write('--------------------------------------------------------------------------------------------------\n')
                    f.write('->MODEL\n')
                    for model in self.model.Model:
                        f.write('   DISCRIMINATOR:\n')
                        [f.write('      ' + x.name + '\n') for x in model.Discriminator.layers]
                        f.write('   GENERATOR:\n')
                        [f.write('      ' + x.name + '\n') for x in model.Generator.layers]
                    f.write('==================================================================================================\n')

        else:
            training, analysis, architecture = update_log(self.parameters,self.model)
            case_ID = self.parameters.analysis['case_ID']
            storage_folder = os.path.join(self.case_dir,'Results',str(case_ID))
            with open(os.path.join(storage_folder,'Model','CGGAN.log'),'w') as f:
                f.write('CGGAN log file\n')
                f.write(
                    '==================================================================================================\n')
                f.write('->ANALYSIS\n')
                for item in analysis.items():
                    f.write(item[0] + '=' + str(item[1]) + '\n')
                f.write(
                    '--------------------------------------------------------------------------------------------------\n')
                f.write('->TRAINING\n')
                for item in training.items():
                    f.write(item[0] + '=' + str(item[1]) + '\n')
                f.write(
                    '--------------------------------------------------------------------------------------------------\n')
                f.write('->ARCHITECTURE\n')
                for item in architecture.items():
                    f.write(item[0] + '=' + str(item[1]) + '\n')
                f.write(
                    '--------------------------------------------------------------------------------------------------\n')
                f.write('->MODEL\n')
                for model in self.model.Model:
                    f.write('   DISCRIMINATOR:\n')
                    [f.write('      ' + x.name + '\n') for x in model.Discriminator.layers]
                    f.write('   GENERATOR:\n')
                    [f.write('      ' + x.name + '\n') for x in model.Generator.layers]
                f.write(
                    '==================================================================================================\n')

if __name__ == '__main__':
    launcher = r'C:\Users\juan.ramos\CGGAN\Scripts\launcher.dat'
    trainer = CGGAN(launcher)
    trainer.launch_analysis()