# -*- coding: utf-8 -*-
import os
from shutil import rmtree, copytree
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
import pickle
import cv2 as cv
from random import randint

import tensorflow as tf
#from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()

import reader
import dataset_processing
import models
import dataset_augmentation
import postprocessing


class CGenTrainer:

    def __init__(self, launch_file):

        self.parameters = self.create_container()
        self.datasets = self.create_container()
        self.datasets.iterators = self.create_container()
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
            model, history = self.reconstruct_model()
            self.model.Model = [model]
            self.model.History = [history]
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

        dataset_train, train_iterator, dataset_cv, cv_iterator,dataset_test = dataset_processing.get_tensorflow_datasets(self.datasets.data_train,self.datasets.data_cv,
                                                                                self.datasets.data_test,batch_size)
        self.datasets.dataset_train = dataset_train
        self.datasets.dataset_cv = dataset_cv
        self.datasets.dataset_test = dataset_test
        self.datasets.iterators.train_iterator = train_iterator
        self.datasets.iterators.cv_iterator = cv_iterator

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
        img_dims = self.parameters.img_size
        latent_dim = self.parameters.training_parameters['latent_dim']
        batch_size = self.parameters.training_parameters['batch_size']
        training_size = self.parameters.training_parameters['train_size']
        n = self.parameters.activation_plotting['n_samples']
        case_ID = self.parameters.analysis['case_ID']
        figs_per_row = self.parameters.activation_plotting['n_cols']
        rows_to_cols_ratio = self.parameters.activation_plotting['rows2cols_ratio']

        # Generate datasets
        self.datasets.data_train, self.datasets.data_cv, self.datasets.data_test = \
        dataset_processing.get_datasets(case_dir,training_size,img_dims)
        self.datasets.dataset_train, self.datasets.dataset_cv, self.datasets.dataset_test = \
        dataset_processing.get_tensorflow_datasets(self.datasets.data_train,self.datasets.data_cv,self.datasets.data_test,batch_size)

        m_tr = self.datasets.data_train[0].shape[0]
        m_cv = self.datasets.data_cv[0].shape[0]
        m_ts = self.datasets.data_test[0].shape[0]
        m = m_tr + m_cv + m_ts

        # Read datasets
        dataset = np.zeros((m,np.prod(img_dims)),dtype='uint8')
        dataset[:m_tr,:] = self.datasets.data_train[0]
        dataset[m_tr:m_tr+m_cv,:] = self.datasets.data_cv[0]
        dataset[m_tr+m_cv:m,:] = self.datasets.data_test[0]

        # Index image sampling
        idx = [randint(1,m) for i in range(n)]
        idx_set = set(idx)
        while len(idx) != len(idx_set):
            extra_item = randint(1,m)
            idx_set.add(extra_item)

        # Reconstruct encoder model
        encoder = self.reconstruct_encoder_CNN()

        # Plot
        for idx in idx_set:
            img = dataset[idx,:]
            postprocessing.monitor_hidden_layers(img,encoder,case_dir,figs_per_row,rows_to_cols_ratio,idx)

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
        n_samples = self.parameters.samples_generation['n_samples']
        training_size = casedata.training_parameters['train_size']
        img_size = casedata.img_size

        if self.model.imported == False:
            self.singletraining()

        if not hasattr(self, 'data_train'):
            data_train, data_cv, data_test = dataset_processing.get_datasets(case_dir,training_size,img_size)
            for model in self.model.Model:
                postprocessing.plot_dataset_samples(data_train,model.predict,n_samples,img_size,storage_dir,stage='Train')
                postprocessing.plot_dataset_samples(data_cv,model.predict,n_samples,img_size,storage_dir,stage='Cross-validation')
                postprocessing.plot_dataset_samples(data_test,model.predict,n_samples,img_size,storage_dir,stage='Test')

        ## GENERATE NEW DATA - SAMPLING ##
        X_samples = self.generate_samples(casedata)
        postprocessing.plot_generated_samples(X_samples,img_size,storage_dir)

    def train_model(self, sens_var=None):

        # Parameters
        case_ID = self.parameters.analysis['case_ID']
        image_shape = (self.parameters.img_size[1],self.parameters.img_size[0],1)
        noise_dim = self.parameters.training_parameters['noise_dim']
        alpha = self.parameters.training_parameters['learning_rate']
        nepoch = self.parameters.training_parameters['epochs']
        epoch_iter = self.parameters.training_parameters['epoch_iter']
        num_iter = nepoch * epoch_iter
        batch_size = self.parameters.training_parameters['batch_size']
        batch_shape = (batch_size,image_shape[1],image_shape[0],1)
        l2_reg = self.parameters.training_parameters['l2_reg']
        l1_reg = self.parameters.training_parameters['l1_reg']
        dropout = self.parameters.training_parameters['dropout']
        activation = self.parameters.training_parameters['activation']

        # Create model containers
        if sens_var != None:
            self.model.Model = []
            history = self.create_container()
            self.model.History = []
        else:
            self.model.Model = self.create_container()
            self.model.History = self.create_container()

        # Training variables
        self.model.History.disc_loss_train = np.zeros([nepoch,])
        self.model.History.disc_metric_train = np.zeros([nepoch,])
        self.model.History.gen_loss_train = np.zeros([nepoch,])
        self.model.History.gen_metric_train = np.zeros([nepoch,])
        # Validation variables
        self.model.History.disc_loss_cv = np.zeros([nepoch,])
        self.model.History.disc_metric_cv = np.zeros([nepoch,])
        self.model.History.gen_loss_cv = np.zeros([nepoch,])
        self.model.History.gen_metric_cv = np.zeros([nepoch,])

        # Models and functions declaration
        discriminator = models.Discriminator(activation,l2_reg,l1_reg,dropout)
        generator = models.Generator(activation,l2_reg,l1_reg,dropout)
        disc_optimizer = models.optimizer(alpha)
        gen_optimizer = models.optimizer(alpha)
        loss = models.loss_function()
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
        for i in range(1,num_iter + 1):
            ### Update discriminator
            with tf.GradientTape() as tape_disc:
                real_image_batch = tf.reshape(self.datasets.iterators.train_iterator.get_next(),batch_shape)
                noise_batch = tf.random.normal((batch_size,noise_dim))
                fake_image_batch = generator(noise_batch)
                # Prediction computations
                fake_logit_batch = discriminator(fake_image_batch)
                real_logit_batch = discriminator(real_image_batch)
                # Ground truth labels
                fake_label_batch = tf.zeros((batch_size,1))
                real_label_batch = tf.ones((batch_size,1))
                # Loss computation
                fake_loss_batch = loss(fake_logit_batch,fake_label_batch)
                real_loss_batch = loss(real_logit_batch,real_label_batch)
                disc_loss_batch = 0.5 * (fake_loss_batch + real_loss_batch)
                # Metric computation
                fake_metric_batch = metric_disc(fake_logit_batch,fake_label_batch).numpy()
                real_metric_batch = metric_disc(real_logit_batch,real_label_batch).numpy()
                disc_metric_batch = 0.5 * (fake_metric_batch + real_metric_batch)
            # Weights update
            disc_gradients = tape_disc.gradient(disc_loss_batch,discriminator.trainable_variables)
            disc_optimizer.apply_gradients(zip(disc_gradients,discriminator.trainable_variables))

            ### Update generator
            with tf.GradientTape() as tape_gen:
                noise_batch = tf.random.normal((batch_size,noise_dim))
                fake_image_batch = generator(noise_batch)
                fake_logit_batch = discriminator(fake_image_batch)
                fake_label_batch = tf.ones((batch_size,1))
                gen_loss_batch = loss(fake_logit_batch,fake_label_batch)
                gen_metric_batch = metric_disc(fake_logit_batch,fake_label_batch).numpy()
            gen_gradients = tape_gen.gradient(gen_loss_batch,generator.trainable_variables)
            gen_optimizer.apply_gradients(zip(gen_gradients,generator.trainable_variables))

            disc_streaming_loss += disc_loss_batch
            disc_streaming_metric += disc_metric_batch
            gen_streaming_loss += gen_loss_batch
            gen_streaming_metric += gen_metric_batch

            if i % epoch_iter == 0:
                # Evaluate on training dataset
                self.model.History.disc_loss_train[epoch] = disc_streaming_loss/epoch_iter
                self.model.History.disc_metric_train[epoch] = disc_streaming_metric/epoch_iter
                self.model.History.gen_loss_train[epoch] = gen_streaming_loss/epoch_iter
                self.model.History.gen_metric_train[epoch] = gen_streaming_metric/epoch_iter
                # Evaluate on cross-validation dataset
                while True:
                    try:
                        ## Evaluate discriminator
                        real_image_cv = tf.reshape(self.datasets.iterators.cv_iterator.get_next(),1)
                        noise_cv = tf.random.normal((1,noise_dim))
                        fake_image_cv = generator(noise_cv,activation,l2_reg,l1_reg,dropout)
                        # Prediction computations
                        fake_logit_cv = discriminator(fake_image_cv,activation,l2_reg,l1_reg,dropout)
                        real_logit_cv = discriminator(real_image_cv,activation,l2_reg,l1_reg,dropout)
                        # Ground truth labels
                        fake_label_cv = tf.zeros((1,1))
                        real_label_cv = tf.ones((1,1))
                        # Loss computation
                        fake_loss_cv = loss(fake_logit_cv,fake_label_cv)
                        real_loss_cv = loss(real_logit_cv,real_label_cv)
                        disc_loss_cv = 0.5 * (fake_loss_cv + real_loss_cv)
                        # Metric computation
                        fake_metric_cv = metric_disc(fake_logit_cv,fake_label_cv).numpy()
                        real_metric_cv = metric_disc(real_logit_cv,real_label_cv).numpy()
                        disc_metric_cv = 0.5 * (fake_metric_cv + real_metric_cv)

                        ## Evaluate generator
                        noise_cv = tf.random.normal(1,noise_dim)
                        fake_image_cv = generator(noise_cv,activation,l2_reg,l1_reg,dropout)
                        fake_logit_cv = discriminator(fake_image_cv,activation,l2_reg,l1_reg,dropout)
                        fake_label_cv = tf.ones((1,1))
                        gen_loss_cv = loss(fake_logit_cv,fake_label_cv)
                        gen_metric_cv = metric_disc(fake_logit_cv,fake_label_cv).numpy()

                        disc_streaming_loss_cv += disc_loss_cv
                        disc_streaming_metric_cv += disc_metric_cv
                        gen_streaming_loss_cv += gen_loss_cv
                        gen_streaming_metric_cv += gen_metric_cv
                        niter += 1
                    except tf.errors.OutOfRangeError:
                        self.model.History.disc_loss_cv[epoch] = disc_streaming_loss_cv/niter
                        self.model.History.disc_metric_cv[epoch] = disc_streaming_metric_cv/niter
                        self.model.History.gen_loss_cv[epoch] = gen_streaming_loss_cv/niter
                        self.model.History.gen_metric_cv[epoch] = gen_streaming_metric_cv/niter
                        niter = 0
                        break

                # Print results
                print('Epoch {}, Discriminator loss (T,CV): ({:.2f},{:.2f}), Discriminator accuracy (T,CV): ({:.2f},{:.2f}) ||'
                      'Generator loss (T,CV): ({:.2f},{:.2f}), Generator accuracy (T,CV): ({:.2f},{:.2f})'
                      .format(epoch,
                              self.model.History.disc_loss_train[epoch],
                              self.model.History.disc_loss_cv[epoch],
                              self.model.History.disc_metric_train[epoch],
                              self.model.History.disc_metric_cv[epoch],
                              self.model.History.gen_loss_train[epoch],
                              self.model.History.gen_loss_cv[epoch],
                              self.model.History.gen_metric_train[epoch],
                              self.model.History.gen_metric_cv[epoch],))

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

    def generate_samples(self, parameters):

        ## BUILD DECODER ##
        output_dim = parameters.img_size
        latent_dim = parameters.training_parameters['latent_dim']
        alpha = parameters.training_parameters['learning_rate']
        dec_hidden_layers = parameters.training_parameters['dec_hidden_layers']
        activation = parameters.training_parameters['activation']
        architecture = parameters.training_parameters['architecture']
        training_size = parameters.training_parameters['train_size']
        batch_size = parameters.training_parameters['batch_size']
        n_samples = self.parameters.samples_generation['n_samples']
        
        decoder = models.VAE(output_dim,latent_dim,[],dec_hidden_layers,alpha,0.0,0.0,0.0,activation,'sample',architecture)  # No regularization
        
        X_samples = []
        for model in self.model.Model:
            # Retrieve decoder weights
            j = 0
            for layer in model.layers:
                if layer.name.startswith('decoder') == False:
                    j += len(layer.weights)
                else:
                    break
            decoder_input_layer_idx = j

            decoder_weights = model.get_weights()[decoder_input_layer_idx:]
            decoder.set_weights(decoder_weights)

            ## SAMPLE IMAGES ##
            samples = np.zeros([n_samples,np.prod(output_dim)])
            for i in range(n_samples):
                t = tf.random.normal(shape=(1,latent_dim))
                samples[i,:] = decoder.predict(t,steps=1)
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

            Nepochs = self.parameters.training_parameters['epochs']
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
                # Export discriminator logs
                loss_filepath = os.path.join(results_folder,'CGGAN_discriminator_loss.dat')
                with open(loss_filepath,'w') as f:
                    if not os.path.isfile(loss_filepath):
                        f.write('Epoch{}Training{}CV\n'.format(delimiter,delimiter))
                    for i in range(nepoch):
                        f.write('{:d}{}{:.2f}{}{:.2f}\n').format(i+1,delimiter,disc_loss_train[i],delimiter,disc_loss_cv[i])
                metrics_filepath = os.path.join(results_folder,'CGGAN_discriminator_metrics.dat')
                with open(metrics_filepath,'w') as f:
                    if not os.path.isfile(metrics_filepath):
                        f.write('Epoch{}Training{}CV\n'.format(delimiter,delimiter))
                    for i in range(nepoch):
                        f.write('{:d}{}{:.2f}{}{:.2f}\n').format(i+1,delimiter,disc_metric_train[i],delimiter,disc_metric_cv[i])
                # Export generator logs
                loss_filepath = os.path.join(results_folder,'CGGAN_generator_loss.csv')
                with open(loss_filepath,'w') as f:
                    if not os.path.isfile(loss_filepath):
                        f.write('Epoch{}Training{}CV\n'.format(delimiter,delimiter))
                    for i in range(nepoch):
                        f.write('{:d}{}{:.2f}{}{:.2f}\n').format(i+1,delimiter,gen_loss_train[i],delimiter,gen_loss_cv[i])
                metrics_filepath = os.path.join(results_folder,'CGGAN_generator_metrics.dat')
                with open(metrics_filepath,'w') as f:
                    if not os.path.isfile(metrics_filepath):
                        f.write('Epoch{}Training{}CV\n'.format(delimiter,delimiter))
                    for i in range(nepoch):
                        f.write('{:d}{}{:.2f}{}{:.2f}\n').format(i+1,delimiter,gen_metric_train[i],delimiter,gen_metric_cv[i])

                ## PLOTS ##
                # Discriminator
                fig_disc, ax_disc = plt.subplots(1)
                ax_disc.plot(epochs,disc_loss_train,label='Training',color='r')
                ax_disc.plot(epochs,disc_loss_cv,label='Cross-validation',color='b')
                ax_disc.grid()
                ax_disc.set_xlabel('Epochs',size=12)
                ax_disc.set_ylabel('Loss',size=12)
                ax_disc.tick_params('both',labelsize=10)
                ax_disc.legend()
                plt.suptitle('Discriminator loss evolution case = {}'.format(str(case_ID)))

                # Generator
                fig_gen, ax_gen = plt.subplots(1)
                ax_gen.plot(epochs,gen_loss_train,label='Training',color='r')
                ax_gen.plot(epochs,gen_loss_cv,label='Cross-validation',color='b')
                ax_gen.grid()
                ax_gen.set_xlabel('Epochs',size=12)
                ax_gen.set_ylabel('Loss',size=12)
                ax_gen.tick_params('both',labelsize=10)
                ax_gen.legend()
                plt.suptitle('Generator loss evolution case = {}'.format(str(case_ID)))

                if sens_var:
                    if type(sens_var[1][i]) == str:
                        results_folder = os.path.join(results_folder,'{}={}'.format(sens_var[0],sens_var[1][i]))
                    else:
                        results_folder = os.path.join(results_folder,'{}={:.3f}'.format(sens_var[0],sens_var[1][i]))
                    os.mkdir(results_folder)
                    disc_loss_plot_filename = 'Discriminator_loss_evolution_{}_{}={}.png'.format(str(case_ID),sens_var[0],str(sens_var[1][i]))
                    gen_loss_plot_filename = 'Generator_loss_evolution_{}_{}={}.png'.format(str(case_ID),sens_var[0],str(sens_var[1][i]))
                else:
                    disc_loss_plot_filename = 'Discriminator_loss_evolution_{}.png'.format(str(case_ID))
                    gen_loss_plot_filename = 'Generator_loss_evolution_{}.png'.format(str(case_ID))

                fig_disc.savefig(os.path.join(results_folder,disc_loss_plot_filename),dpi=200)
                fig_gen.savefig(os.path.join(results_folder,gen_loss_plot_filename),dpi=200)
                plt.close('all')

    def export_model(self, sens_var=None):

        N = len(self.model.Model)
        case_ID = self.parameters.analysis['case_ID']
        for i in range(N):
            if sens_var:
                if type(sens_var[1][i]) == str:
                    storage_dir = os.path.join(self.case_dir,'Results',str(case_ID),'Model','{}={}'
                                               .format(sens_var[0],sens_var[1][i]))
                else:
                    storage_dir = os.path.join(self.case_dir,'Results',str(case_ID),'Model','{}={:.3f}'
                                               .format(sens_var[0],sens_var[1][i]))
                model_json_name = 'CGGAN_model_{}_{}={}_arquitecture.json'.format(str(case_ID),sens_var[0],str(sens_var[1][i]))
                model_weights_name = 'CGGAN_model_{}_{}={}_weights.h5'.format(str(case_ID),sens_var[0],str(sens_var[1][i]))
                model_folder_name = 'CGGAN_model_{}_{}={}'.format(str(case_ID),sens_var[0],str(sens_var[1][i]))
            else:
                storage_dir = os.path.join(self.case_dir,'Results',str(case_ID),'Model')
                model_json_name = 'CGGAN_model_{}_arquitecture.json'.format(str(case_ID))
                model_weights_name = 'CGGAN_model_{}_weights.h5'.format(str(case_ID))
                model_folder_name = 'CGGAN_model_{}'.format(str(case_ID))

            if os.path.exists(storage_dir):
                rmtree(storage_dir)
            os.makedirs(storage_dir)

            # Export history training
            with open(os.path.join(storage_dir,'History'),'wb') as f:
                pickle.dump(self.model.History[i].history,f)

            # Save model
            # Export model arquitecture to JSON file
            model_json = self.model.Model[i].to_json()
            with open(os.path.join(storage_dir,model_json_name),'w') as json_file:
                json_file.write(model_json)
            self.model.Model[i].save(os.path.join(storage_dir,model_folder_name.format(str(case_ID))))

            # Export model weights to HDF5 file
            self.model.Model[i].save_weights(os.path.join(storage_dir,model_weights_name))

    def reconstruct_model(self, mode='train'):

        storage_dir = os.path.join(self.case_dir,'Results','pretrained_model')
        try:
            casedata = reader.read_case_logfile(os.path.join(storage_dir,'CGGAN.log'))
            img_dim = casedata.img_size
            latent_dim = casedata.training_parameters['latent_dim']
            enc_hidden_layers = casedata.training_parameters['enc_hidden_layers']
            dec_hidden_layers = casedata.training_parameters['dec_hidden_layers']
            activation = casedata.training_parameters['activation']
            architecture = casedata.training_parameters['architecture']

            # Load weights into new model
            Model = models.VAE(img_dim,latent_dim,enc_hidden_layers,dec_hidden_layers,0.001,0.0,0.0,0.0,activation,
                               mode,architecture)
            weights_filename = [file for file in os.listdir(storage_dir) if file.endswith('.h5')][0]
            Model.load_weights(os.path.join(storage_dir,weights_filename))
            class history_container:
                pass
            History = history_container()
            with open(os.path.join(storage_dir,'History'),'rb') as f:
                History.history = pickle.load(f)
            History.epoch = None
            History.model = Model
        except:
            tf.config.run_functions_eagerly(True) # Enable eager execution
            try:
                model_folder = next(os.walk(storage_dir))[1][0]
            except:
                print('There is no model stored in the folder')

            alpha = self.parameters.training_parameters['learning_rate']
            loss = models.loss_function

            Model = tf.keras.models.load_model(os.path.join(storage_dir,model_folder),custom_objects={'loss':loss},compile=False)
            Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha),loss=lambda x, y: loss,
                          metrics=[tf.keras.metrics.MeanSquaredError()])

            tf.config.run_functions_eagerly(False) # Disable eager execution

            # Reconstruct history
            class history_container:
                pass
            History = history_container()
            try:
                with open(os.path.join(storage_dir,'History'),'rb') as f:
                    History.history = pickle.load(f)
                History.epoch = np.arange(1,len(History.history['loss'])+1)
                History.model = Model
            except:
                History.epoch = None
                History.model = None

        return Model, History

    def reconstruct_encoder_CNN(self):

        img_dim = self.parameters.img_size
        latent_dim = self.parameters.training_parameters['latent_dim']
        enc_hidden_layers = self.parameters.training_parameters['enc_hidden_layers']
        dec_hidden_layers = self.parameters.training_parameters['dec_hidden_layers']
        activation = self.parameters.training_parameters['activation']
        architecture = self.parameters.training_parameters['architecture']

        storage_dir = os.path.join(self.case_dir,'Results','pretrained_model')

        if architecture == 'cnn':
            Encoder = models.encoder_lenet(img_dim,latent_dim,enc_hidden_layers,0.0,0.0,0.0,activation)
        else:
            Encoder = models.encoder(np.prod(img_dim),enc_hidden_layers,latent_dim,activation)
        Encoder.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.MeanSquaredError())

        # Load weights into new model
        Model = models.VAE(img_dim,latent_dim,enc_hidden_layers,dec_hidden_layers,0.001,0.0,0.0,0.0,'relu','train',architecture)
        Model.load_weights(os.path.join(storage_dir,'CGGAN_model_weights.h5'))
        enc_CNN_last_layer_idx = [idx for (idx,weight) in enumerate(Model.weights) if weight.shape[0] == latent_dim][0]
        encoder_weights = Model.get_weights()[:enc_CNN_last_layer_idx]
        Encoder.set_weights(encoder_weights)

        return Encoder

    def export_nn_log(self):
        def update_log(parameters, model):
            training = OrderedDict()
            training['ARCHITECTURE'] = parameters.training_parameters['architecture']
            training['TRAINING SIZE'] = parameters.training_parameters['train_size']
            training['LEARNING RATE'] = parameters.training_parameters['learning_rate']
            training['L2 REGULARIZER'] = parameters.training_parameters['l2_reg']
            training['L1 REGULARIZER'] = parameters.training_parameters['l1_reg']
            training['DROPOUT'] = parameters.training_parameters['dropout']
            training['ACTIVATION'] = parameters.training_parameters['activation']
            training['NUMBER OF EPOCHS'] = parameters.training_parameters['epochs']
            training['BATCH SIZE'] = parameters.training_parameters['batch_size']
            training['LATENT DIMENSION'] = parameters.training_parameters['latent_dim']
            training['ENCODER HIDDEN LAYERS'] = parameters.training_parameters['enc_hidden_layers']
            training['DECODER HIDDEN LAYERS'] = parameters.training_parameters['dec_hidden_layers']
            training['OPTIMIZER'] = [model.optimizer._name for model in model.Model]
            training['METRICS'] = [model.metrics_names[-1] if model.metrics_names != None else None for model in model.Model]

            analysis = OrderedDict()
            analysis['CASE ID'] = parameters.analysis['case_ID']
            analysis['ANALYSIS'] = parameters.analysis['type']
            analysis['IMPORTED MODEL'] = parameters.analysis['import']
            analysis['LAST TRAINING LOSS'] = ['{:.3f}'.format(history.history['loss'][-1]) for history in model.History]
            analysis['LAST CV LOSS'] = ['{:.3f}'.format(history.history['val_loss'][-1]) for history in model.History]

            architecture = OrderedDict()
            architecture['INPUT SHAPE'] = parameters.img_size

            return training, analysis, architecture


        parameters = self.parameters
        if parameters.analysis['type'] == 'sensanalysis':
            varname, varvalues = parameters.sens_variable
            for value in varvalues:
                parameters.training_parameters[varname] = value
                training, analysis, architecture = update_log(parameters,self.model)

                case_ID = parameters.analysis['case_ID']
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
                        model.summary(print_fn=lambda x: f.write(x + '\n'))
                    f.write('==================================================================================================\n')

        else:
            training, analysis, architecture = update_log(self.parameters,self.model)
            case_ID = parameters.analysis['case_ID']
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
                    model.summary(print_fn=lambda x: f.write(x + '\n'))
                f.write(
                    '==================================================================================================\n')
if __name__ == '__main__':
    launcher = r'C:\Users\juan.ramos\CGGAN\Scripts\launcher.dat'
    trainer = CGenTrainer(launcher)
    trainer.launch_analysis()