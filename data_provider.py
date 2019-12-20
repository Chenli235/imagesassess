# -*- coding: utf-8 -*-

import tensorflow 
import numpy 
import tensorflow.contrib.slim
import os


IMAGE_WIDTH = 520
IMAGE_HEIGHT = 520
FEATURE_IMAGE = 'image'
FEATURE_IMAGE_CLASS = "image/class"
FEATURE_IMAGE_PATH = "image/path" 

_ITEMS_TO_DESCRIPTIONS = {
        FEATURE_IMAGE : 'A [width x width x 1] grayscale image.',
        FEATURE_IMAGE_CLASS:'A single integer between 0 and [num_classes-1]',
        FEATURE_IMAGE_PATH:'A string indication path to image',
        }
# range of random brightness factors to scale training data.
_BRIGHTNESS_MIN_FACTOR = 0.2
_BRIGHTNESS_MAX_FACTOR = 5.0

# range of random image brighteness offsets for training data.
_BRIGHTNESS_MIN_OFFSET = 1.0/65535
_BRIGHTNESS_MAX_OFFSET = 1000.0/65535

def get_filename_num_records(tf_record_path):
    """get path to text file containing number of records.
    Args:
        tf_record_path
    returns:
        string jpath to text file containing number of records in TFRecord file.
    """
    return os.path.splitext(tf_record_path)[0] + '.num_records'

def get_num_records(tf_record_path):
    num_records_path = get_filename_num_records(tf_record_path)
    with open(num_records_path, 'r') as f:
        num_records = int(f.read())
        return num_records

def get_split(split_name,tfrecord_file_pattern,num_classes,image_width,image_height):
    """gets a dataset tuple from tfrecord, to be used with datasetdataprovider
    
    Args:
        split_name: string, a train/test split name
        tfrecord_file_pattern:string with formating for split name E.g. 'file_s.tfrecord'
        num_classes: integer representing number of classes.
        image_width,image_height
    return:
        A Dataset namedtuple
    """    
    valid_splits = {'train','test'}
    if split_name not in valid_splits:
        raise ValueError('split name %s was not recognized.' %split_name)
    
    if image_height <= 0 or image_width <= 0:
        raise ValueError('Invalid image_height and/or image_width: %d, %d.' %
                         image_height, image_width)
    image_shape = (image_height, image_width, 1)
    keys_to_features = {
            FEATURE_IMAGE:
                tensorflow.FixedLenFeature(image_shape,tensorflow.float32,default_value=tensorflow.zeros(image_shape)),
            FEATURE_IMAGE_CLASS:
                tensorflow.FixedLenFeature([num_classes],tensorflow.float32,default_value=tensorflow.zeros([num_classes])),
            FEATURE_IMAGE_PATH:
                tensorflow.FixedLenFeature([1],tensorflow.string,default_value='')
                }
    items_to_handlers={
            FEATURE_IMAGE:tensorflow.contrib.slim.tfexample_decoder.Tensor(FEATURE_IMAGE),
            FEATURE_IMAGE_CLASS:tensorflow.contrib.slim.tfexample_decoder.Tensor(FEATURE_IMAGE_CLASS),
            FEATURE_IMAGE_PATH:tensorflow.contrib.slim.tfexample_decoder.Tensor(FEATURE_IMAGE_PATH)
                }
    
    decoder = tensorflow.contrib.slim.tfexample_decoder.TFExampleDecoder(keys_to_features,items_to_handlers)
    file_pattern=tfrecord_file_pattern % split_name
    num_samples = get_num_records(file_pattern)
    return tensorflow.contrib.slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=tensorflow.TFRecordReader,
            decoder=decoder,
            num_samples=num_samples,
            num_classes=num_classes,
            items_to_descriptions=_ITEMS_TO_DESCRIPTIONS)

def get_batches(image,label,image_path,num_threads=800,batch_size=32):
    
    """converts images and lables into baatches.
        args:
            image:input image tensor, size [num_images * width*heights*1]
            labels: input lable tensor, size [num_images * 1]
            image_path: input image path tensor, size [num_images * 1]
            num_threads:Integer, number of threads for preprocessing and loading data.
            batch size: integer
        return:
            batched version of the inputs: images(shape [batch_size*width*width*1],
            labels(shape [batch_size*num_classes]) and image_paths(shape [batch_size * 1])
    """
    assert len(image.get_shape().as_list()) == 4
    batch_images,batch_one_hot_labels,batch_image_paths = tensorflow.train.batch(
                [image,label,image_path],
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=5*batch_size,
                enqueue_many=True)
    return batch_images,batch_one_hot_labels,batch_image_paths
    
def get_image_patch_tensor(image, label, image_path, patch_width):
    """Crops a random patch from image.
    Args:
        image: Input image tensor, size [width x width x 1].
        label: Input label tensor, size [num_classes].
        image_path: Input image path tensor, size [1].
        patch_width: Integer representing width of image patch.
    Returns:
        Tensors image patch, size [1 x patch_width x patch_width x 1],
        expanded_label, size [1 x num_classes], and expanded_image_path, size
        [1 x 1].
    """
    assert len(image.get_shape().as_list()) == 3, image.get_shape().as_list()
    size = tensorflow.constant([patch_width, patch_width, 1], dtype=tensorflow.int32)
    patch = tensorflow.expand_dims(tensorflow.random_crop(image, size), 0)
    expanded_label = tensorflow.expand_dims(label, dim=0)
    expanded_image_path = tensorflow.expand_dims(image_path, dim=0)
    return patch,expanded_label,expanded_image_path

def apply_random_offset(patch,min_offset,max_offset):
    """Adds a random offset to input image(tensor)."""
    # Choose offset uniformly in log space.
    offset = tensorflow.pow(tensorflow.constant([10.0]),tensorflow.random_uniform([1],numpy.log10(min_offset),numpy.log10(max_offset)))
    return tensorflow.add(patch,offset)

def apply_random_brightness_adjust(patch, min_factor, max_factor):
    """Scales the input image (tensor) brightness by a random factor."""
    # Choose brightness scale uniformly in log space.
    brightness = tensorflow.pow(
        tensorflow.constant([10.0]),
        tensorflow.random_uniform([1], numpy.log10(min_factor), numpy.log10(max_factor)))
    return tensorflow.multiply(patch, brightness)

def get_image_tiles_tensor(image, label, image_path, patch_width):    
    """Gets patches that tile the input image, starting at upper left.
    args:
        image: input image tensor
        label: input label tensor
        image_path: Input image path tensor
        patch_width: Integer representing width of image patch
    returns:
        tensors tiles, size[num_tiles*patch_width*patch_width*1],labels,
        size [num_tiles*number_classes], and image_paths,size [num_tiles *1]
    """
    tiles_before_reshape = tensorflow.extract_image_patches(
            tensorflow.expand_dims(image,dim=0),[1,patch_width,patch_width,1],
            [1,patch_width,patch_width,1],[1,1,1,1],'VALID')
    tiles = tensorflow.reshape(tiles_before_reshape,[-1,patch_width,patch_width,1])
    
    labels = tensorflow.tile(tensorflow.expand_dims(label,dim=0),[tensorflow.shape(tiles)[0],1])
    image_paths = tensorflow.tile(tensorflow.expand_dims(image_path,dim=0),[tensorflow.shape(tiles)[0],1])
    
    return tiles,labels,image_paths

def provide_data(tfrecord_file_pattern,
                 split_name,
                 batch_size,
                 num_classes,
                 image_width,
                 image_height,
                 patch_width=28,
                 randomize=True,
                 num_threads=64):
    
    """Provides batches of data. 
      Args:
    tfrecord_file_pattern: String, with formatting for split name. E.g.
      'file_%s.tfrecord'.
    split_name: String indicating split name, typically 'train' or 'test'.
    batch_size: The number of images in each batch. If 'randomize' is False,
      the batch size must be the number of tiles per image.
    num_classes: Integer representing number of classes.
    image_width: Integer, width of image size to be cropped.
    image_height: Integer, height of image size to be cropped.
    patch_width: Integer width (and height) of image patch.
    randomize: Boolean indicating whether to use image patches that are randomly
      cropped, with a random offset and brightness adjustment applied. Use only
      for training.
    num_threads: Number of threads for data reading queue. Use only 1 thread for
      deterministic ordering of inputs.
  Returns:
    batch_images: A `Tensor` of size [batch_size, patch_width, patch_width, 1]
    batch_one_hot_labels: A `Tensor` of size [batch_size, num_classes], where
      each row has a single element set to one and the rest set to zeros.
    num_samples: The number of images (not tiles) in the dataset.
  Raises:
    ValueError: If the batch size is invalid.
    """
    if batch_size <= 0:
       raise ValueError('Invalid batch size: %d' % batch_size)

    dataset_info = get_split(
            split_name,
            tfrecord_file_pattern,
            num_classes,
            image_width = image_width,
            image_height = image_height)
    provider = tensorflow.contrib.slim.dataset_data_provider.DatasetDataProvider(
            dataset_info,
            common_queue_capacity=2*batch_size,
            common_queue_min=batch_size,
            shuffle=False,
            num_readers = num_threads)
    [image, label, image_path] = provider.get(
        [FEATURE_IMAGE, FEATURE_IMAGE_CLASS, FEATURE_IMAGE_PATH])
   
    if randomize:
       # For training, get a single randomly cropped image patch.
        patch_original, label, image_path = get_image_patch_tensor(
            image, label, image_path, patch_width=patch_width)
        
        # Apply a random offset and brightness adjustment.
        patch_scaled = apply_random_brightness_adjust(
            patch_original,
            min_factor=_BRIGHTNESS_MIN_FACTOR,
            max_factor=_BRIGHTNESS_MAX_FACTOR)
        
        patch = apply_random_offset(
            patch_scaled,
            min_offset=_BRIGHTNESS_MIN_OFFSET,
            max_offset=_BRIGHTNESS_MAX_OFFSET)
        
        batch_images, batch_one_hot_labels, batch_image_paths = get_batches(
            patch,
            label,
            image_path,
            batch_size=batch_size,
            num_threads=num_threads)
        
    else:
        # For testing extract tiles that perfectly tile (without overlap) the image.
        tiles, labels, image_paths = get_image_tiles_tensor(
            image, label, image_path, patch_width=patch_width)

        num_tiles = tiles.get_shape().as_list()[0]
        assert num_tiles == batch_size, 'num_tiles: %d, batch_size: %d' % (
            num_tiles, batch_size)
        batch_images, batch_one_hot_labels, batch_image_paths = get_batches(
            tiles,
            labels,
            image_paths,
            batch_size=num_tiles,
            num_threads=num_threads)
    num_samples = provider.num_samples()
    return batch_images, batch_one_hot_labels, batch_image_paths, num_samples    
