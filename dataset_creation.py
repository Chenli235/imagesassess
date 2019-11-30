# -*- coding: utf-8 -*-

""" Functions for reading data and saving as tfexamples in tfrecord format."""

import collections
import glob
import os

import numpy
import skimage.io
import tensorflow
import tensorflow.core.example
import data_provider

# Threshold for foreground objects (after background subtraction).
_FOREGROUND_THRESHOLD = 100.0 / 65535
# Minimum fraction of nonzero pixels to be considered foreground image.
_FOREGROUND_AREA_THRESHOLD = 0.0001
# Somewhat arbitrary mean foreground value all images are normalized to.
_FOREGROUND_MEAN = 200.0 / 65535

_SUPPORTED_EXTENSIONS = ['.tif', '.tiff', '.png']

class Dataset(object):
    """Holds the image data before training examples are created.
    The actual images are only read when samples are retrieved.
    Attributes:
    labels: Float32 numpy array [num_images x num_classes].
    image_paths: List of image paths (strings).
    num_examples: Integer, number of examples in dataset.
    image_background_value: Float, background value of images in dataset.
    image_brightness_scale: Float, multiplicative exposure factor.
    image_width: Integer, width of image size to be cropped.
    image_height: Integer, height of image size to be cropped.
    """
    
    def __init__(self,
                 labels,
                 image_paths,
                 image_width,
                 image_height,
                 image_background_value=0.0,
                 image_brightness_scale=1.0):
        assert len(labels.shape) == 2
        assert len(image_paths) == labels.shape[0]
        assert image_background_value < 1.0
        assert 0 < image_brightness_scale
        
        self.labels = labels
        self.image_paths = image_paths
        self.num_examples = len(image_paths)
        self.image_background_value = image_background_value
        self.image_brightness_scale = image_brightness_scale
        self.image_width = image_width
        self.image_height = image_height
        assert self.num_examples > 0
        self.subsampled = False
        
    def randomize(self):
        """Randomize the ordering of images and labels."""
        ordering = numpy.random.permutation(range(self.num_examples))
        self.labels = self.labels[ordering, :]
        self.image_paths = list(numpy.array(self.image_paths)[ordering])
        
    def subsample_for_shard(self,shard_num,num_shards):
        """Subsample the data based on the shard."""
        self.labels = self.labels[shard_num::num_shards, :]
        self.image_paths = self.image_paths[shard_num::num_shards]
        self.num_examples = len(self.image_paths)
        self.subsampled = True
        
    def get_sample(self,index,normalize):
        """Get a single sample from the dataset.        
        Args:
            index: Integer, index within dataset for the sample.
            normalize: Boolean, whether to brightness normalize the image.
            
        Returns:
            Tuple of image, a 2D numpy float array, label, a 1D numpy array, and
            image_path, a string path to the image.
        
        Raises:
            ValueError: If the image pixel values are invalid.
        """
        assert index < self.num_examples
        image_path = self.image_paths[index]
        label = self.labels[index, :]
        
        # Read image from disk.
        image = get_preprocessed_image(image_path, self.image_background_value,
                                       self.image_brightness_scale,
                                       self.image_width, self.image_height,
                                       normalize)
        
        assert len(image.shape) == 2
        assert image.dtype == numpy.float32
        assert image.shape[0] == self.image_height
        assert image.shape[1] == self.image_width
        
        if numpy.any(numpy.isnan(image)):
            raise ValueError('NaNs found in image from %s' % image_path)
        if numpy.min(image) < 0.0 or numpy.max(image) > 1.0:
            raise ValueError('Image values exceed range [0,1.0]: [%g,%g]' %
                             (numpy.min(image), numpy.max(image)))
            
        return image, label, image_path
        
    def dataset_to_examples_in_tfrecord(list_of_image_globs,
                                    output_directory,
                                    output_tfrecord_filename,
                                    num_classes,
                                    image_width,
                                    image_height,
                                    max_images=100000,
                                    randomize=True,
                                    image_background_value=0.0,
                                    image_brightness_scale=1.0,
                                    shard_num=None,
                                    num_shards=None,
                                    normalize=True,
                                    use_unlabeled_data=False):
        """Reads dataset and saves as TFExamples in a TFRecord.
        Args:
           list_of_image_globs: List of strings, each a glob. If use_unlabeled_data is
          False, the number of globs must equal num_classes (the images for the ith
          glob  will take the true label for class i -- this is used for training
          and evaluation).
          output_directory: String, path to output direcotry.
        output_tfrecord_filename: String, name for output TFRecord.
        num_classes: Integer, number of classes of defocus.
        image_width: Integer, width of image size to be cropped.
        image_height: Integer, height of image size to be cropped.
        max_images: Integer, max number of images to read per class.
        randomize: Boolean, whether to randomly permute the data ordering.
        image_background_value: Float, background value of images in dataset.
        image_brightness_scale: Float, multiplicative exposure factor.
        shard_num: Integer, if sharding, borg task number.
        num_shards: Integer, if sharding, total number of borg tasks
        normalize: Boolean, whether to brightness normalize the image.
        use_unlabeled_data: Boolean, whether there does not exist true labels.
        Returns:
        Number of converted example images.
        Raises:
        ValueError: If the input image directories are invalid.
        """
        # Get the image paths and labels. Patches will be extracted in data_provider.
        if not use_unlabeled_data:
            if len(list_of_image_globs) != num_classes:
                raise ValueError('%d globs specified, but for labeled data, must be %d' %
                             (len(list_of_image_globs), num_classes))
            dataset = read_labeled_dataset(list_of_image_globs, max_images, num_classes,
                                       image_width, image_height,
                                       image_background_value,
                                       image_brightness_scale)
        else:
            dataset = read_unlabeled_dataset(list_of_image_globs, max_images,
                                         num_classes, image_width, image_height,
                                         image_background_value,
                                         image_brightness_scale)
        if dataset.num_examples == 0:
            raise ValueError('No images found from globs.')
            
        # Optionally subsample if sharding.
        if shard_num is not None and num_shards is not None and num_shards > 1:
            dataset.subsample_for_shard(shard_num, num_shards)
        
        # Convert to Examples and write the result to an TFRecord.
        num_examples = convert_to_examples(dataset, output_directory,
                                       output_tfrecord_filename, randomize,
                                       normalize)
        return num_examples
    
    def get_preprocessed_image(path,
                               image_background_value,
                               image_brightness_scale,
                               image_width,
                               image_height,
                               normalize=True):
        """Read the a tif or png image, background subtract, crop and normalize.
        Args:
            path: Path to 16-bit tif or png image to be read.
            image_background_value: Float value, between 0.0 and 1.0, indicating
            background value to be subtracted from all pixels. This value should be
            empirically determined for each camera, and should represent the mean
            pixel value with zero incident photons. Note that many background pixel
            values will be clipped to zero.
            image_brightness_scale: Float value, greater than one, indicating
            value to scale all images by (prior to normalization). This is primarily
            for evaluating the performance of the automatic normalization.
            image_width: Integer, width of image size to be cropped.
            image_height: Integer, height of image size to be cropped.
            normalize: Boolean, whether to normalize the image based on the mean
            foreground pixel values. Note that the noise amplitude will be affected by
            this operation, but until the model is trained on data spanning the
            entire 16-bits of dynamic range, this is the best approach toward
            handling the large dynamic range.
        returns:
            The preprocessed image as a 2D float numpy array.
        Raises:
            ValueError: if image is too small.
        """
        
        image = read_16_bit_greyscale(path)
        # Background subtraction
        image_without_background = numpy.clip((
        (image - image_background_value) * image_brightness_scale), 0.0, 1.0)
        cropped_image = image_without_background[0:image_height,0:image_width]
        if normalize:
            preprocessed_image = normalize_image(cropped_image)
        else:
            preprocessed_image = cropped_image
            
        return preprocessed_image
        
        
        
    
    def read_16_bit_greyscale(path):
        """Reads a 16-bit png or tif into a numpy array.
        Args:
            path: String indicating path to .png or .tif file to read.
        Returns:
            A float32 numpy array of the greyscale image, where [0, 65535] is mapped to [0, 1].
        """
        
        file_extension = os.path.splitext(path)[1]
        assert (file_extension in _SUPPORTED_EXTENSIONS), 'path is %s' % path
        
        greyscale_map = skimage.io.imread(path)
        
        # Normalize to float in range [0, 1]
        assert numpy.max(greyscale_map) <= 65535
        greyscale_map_normalized = greyscale_map.astype(numpy.float32) / 65535
        return greyscale_map_normalized
    
    
        
        
        
        
        
        
        
        
        
    def read_16_bit_greyscale(path):
        
        
        
    def read_labeled_dataset(list_of_globs,
                             max_images,
                             num_classes,
                             image_width,
                             image_height,
                             image_background_value=0.0,
                             image_brightness_scale=0.0):
        """Gets image paths from disk and create one-hot-encoded labels.
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    