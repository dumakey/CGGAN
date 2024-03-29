import os
import numpy as np
import cv2 as cv
import tensorflow as tf
from sklearn.model_selection import train_test_split

from preprocessing import ImageTransformer
 

def preprocess_tf_data(im):

    im_tf = tf.cast(im,tf.float32)
    im_tf = im_tf/255

    return im_tf

def preprocess_data(im):

    im = im.astype(np.float32)
    im = im/255

    return im

def read_dataset(case_folder, dataset_folder='Training', format='png'):

    img_filepaths = []
    for (root, case_dirs, _) in os.walk(os.path.join(case_folder,'Datasets',dataset_folder)):
        for case_dir in case_dirs:
            files = [os.path.join(root,case_dir,file) for file in os.listdir(os.path.join(root,case_dir)) if file.endswith(format)]
            img_filepaths += files

    img_list = []
    for filepath in img_filepaths:
        img = cv.imread(filepath)
        gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img_list.append(gray_img)

    return img_list

def preprocess_image(imgs, new_dims):

    m = len(imgs)
    width, height = new_dims
    imgs_processed = np.zeros((m,height,width),dtype='uint8')
    for i in range(m):
        if imgs[i].shape[0:2] != (height,width):
            img_processed = ImageTransformer.resize(imgs[i],(width,height))
        else:
            img_processed = imgs[i]
        imgs_processed[i,:] = cv.bitwise_not(img_processed)

    return imgs_processed

def get_datasets(case_folder, training_size, img_dims):

    # Read original datasets
    X = read_dataset(case_folder)
    # Resize images, if necessary
    X = preprocess_image(X,img_dims)

    X_train, X_val = train_test_split(X,train_size=training_size,shuffle=True)
    X_cv, X_test = train_test_split(X_val,train_size=0.75,shuffle=True)
    
    return X_train, X_cv, X_test

def create_dataset_pipeline(dataset, is_train=True, num_threads=8, prefetch_buffer=100, batch_size=32):

    dataset_tensor = tf.data.Dataset.from_tensor_slices(dataset)

    if is_train:
        dataset_tensor = dataset_tensor.shuffle(buffer_size=dataset[0].shape[0]).repeat()
    dataset_tensor = dataset_tensor.map(preprocess_tf_data,num_parallel_calls=num_threads)
    dataset_tensor = dataset_tensor.batch(batch_size)
    dataset_tensor = dataset_tensor.prefetch(prefetch_buffer)

    return dataset_tensor

def get_tensorflow_datasets(data_train, data_cv, data_test, batch_size=32):

    dataset_train = create_dataset_pipeline(data_train,is_train=True,batch_size=batch_size)
    dataset_cv = create_dataset_pipeline(data_cv,is_train=False,batch_size=1)
    dataset_test = preprocess_data(data_test)
    
    return dataset_train, dataset_cv, dataset_test
