# -*- coding: utf-8 -*-

import logging
import os
import glob
import numpy
import six
import tensorflow

# Use this backend for producing PNGs without interactive display.
import matplotlib
matplotlib.use('Agg')

import constants
import data_provider
import dataset_creation
import evaluation
#import prediction
import miq
#import summarize
import validation
import prediction

_MAX_IMAGES_TO_VALIDATE = 1e6

def download(output_path):
    if output_path:
        miq.download_model(output_path=output_path)
    else:
        miq.download_model()

def validate(imagepath,width=None,height=None,patch_width=84):
    image_paths = []
    
    
    image_paths = dataset_creation.get_images_from_glob(imagepath,_MAX_IMAGES_TO_VALIDATE)
    
    print('Found {} paths'.format(len(image_paths)))
    
    if len(image_paths) == 0:
        raise ValueError('No images found.')
    
    validation.check_duplicate_image_name(image_paths)
    
    if width is None or height is None:
        height,width = dataset_creation.image_size_from_glob(imagepath,patch_width)
    print(height,width)    
    validation.check_image_dimensions(image_paths, height, width)
    
def predict(imagepath=None,checkpoint=None,output=None,width=None,height=None,patch_width=84,visualize=True):
    if output is None:
        print('Eval directory required.')
    if checkpoint is None:
        checkpoint = miq.DEFAULT_MODEL_PATH
    if imagepath is None:
        print('Must provide image globs list.')
    if not os.path.isdir(output):
        os.makedirs(output)
    images = glob.glob(imagepath)
    
    use_unlabeled_data = True
    # Input images will be cropped to image_height * image_width
    image_size = dataset_creation.image_size_from_glob(images[0], patch_width)
    print(image_size)
    #print(image_size.height,image_size.width)
    if width is not None and height is not None:
        image_width = int(patch_width*numpy.floor(width/patch_width))
        image_height = int(patch_width*numpy.floor(height/patch_width))
        if image_width > image_size.width or image_height > image_size.height:
            raise ValueError('Specified (image_width,image_height)=(%d,%d) exceeds valid dimensions (%d,%d).' 
                             %(image_width,image_height,image_size.width),image_size.height)
    else:
        image_width = image_size.width
        image_height = image_size.height
    # All patches evaluated in a batch correspond to one single input image.
    batch_size = int(image_width*image_height/(patch_width**2))
    
    print('Using batch_size=%d for image_width=%d, image_height=%d, model_patch_width=%d'  %(batch_size, image_width, image_height, patch_width))
    
    tfexamples_tfrecord = prediction.build_tfrecord_from_pngs(images, use_unlabeled_data, 11, output, 0.0, 1.0, 1, 1, image_width, image_height) 
    
    num_samples = data_provider.get_num_records(tfexamples_tfrecord % prediction._SPLIT_NAME)
    
    logging.info('TFRecord has %g samples.',num_samples)
    
    graph = tensorflow.Graph()
    
    with graph.as_default():
        images,one_hot_labels,image_paths,_ = data_provider.provide_data(
            batch_size = batch_size,
            image_height = image_height,
            image_width=image_width,
            num_classes=11,
            num_threads=1,
            patch_width=patch_width,
            randomize=False,
            split_name=prediction._SPLIT_NAME,
            tfrecord_file_pattern = tfexamples_tfrecord)
        model_metrics = evaluation.get_model_and_metrics(
            images=images,
            is_training=False,
            model_id=0,
            num_classes=11,
            one_hot_labels=one_hot_labels)
        prediction.run_model_inference(
            aggregation_method = evaluation.METHOD_AVERAGE,
            image_height = image_height,
            image_paths = image_paths,
            image_width = image_width,
            images = images,
            labels = model_metrics.labels,
            model_ckpt_file=checkpoint,
            num_samples = num_samples,
            num_shards=1,
            output_directory = os.path.join(output,'miq_result_images'),
            patch_width=patch_width,
            probabilities = model_metrics.probabilities,
            shard_num = 1,
            show_plots = visualize
            )
        # Delete TFRecord to save disk space.
        tfrecord_path = tfexamples_tfrecord % prediction._SPLIT_NAME
        
        os.remove(tfrecord_path)
        logging.info('Deleted %s',tfrecord_path)
        
   

if __name__ == '__main__':
    #download('')
    #validate('tests/data/images_for_glob_test/*.tif')
    predict(imagepath='tests/data/images_for_glob_test/00_mcf-z-stacks-03212011_e24_s2_w1241c3e73-1e5a-4121-b7b5-02af37510046.tif',output = 'tests/output/')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    






