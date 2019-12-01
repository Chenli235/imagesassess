# -*- coding: utf-8 -*-

"""
Evaluates a trained Miq model.

Usage:
  Start eval loop, which runs forever, constantly checking for new model
    checkpoints:
    
    microscopeimagequality evaluate --checkpoint <path_to_train_directory> \
      --output <path_to_train_directory> \
     "/focus0/*,/focus1/*,/focus2/*, \
      /focus3/*,/focus4/*,/focus5/*,/focus6/*,/focus7/*,/focus8/*,/focus9/*, \
      /focus10/*" 

  View training progress:
    tensorboard --logdir=<path_to_train_directory>

    In web browser, go to localhost:6006.
"""

import collections
import csv
import os

import PIL.Image
import PIL.ImageDraw
import matplotlib.pyplot
import numpy
import scipy.misc
import scipy.stats
import skimage.io
import tensorflow
import tensorflow.contrib.slim
import tensorflow.python.ops

import miq

_IMAGE_ANNOTATION_MAGNIFICATION_PERCENT = 800
CERTAINTY_NAMES = ['mean', 'max', 'aggregate', 'weighted']
CERTAINTY_TYPES = {i: CERTAINTY_NAMES[i] for i in range(len(CERTAINTY_NAMES))}
BORDER_SIZE = 8

CLASS_ANNOTATION_COLORMAP = 'hsv'

METHOD_AVERAGE = 'average'
METHOD_PRODUCT = 'product'




















