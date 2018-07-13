import numpy as np
import math

from keras import backend as K
import tensorflow as tf

import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, Conv3D, Reshape, Activation, Layer, Concatenate, Lambda
from keras.initializers import normal, zeros, Initializer
from keras.optimizers import Adam, SGD
from keras.applications.resnet50 import ResNet50
from ml.generators.mip_generator import MipGenerator
from ml.models.FeaturePyramidNetwork import FPNBuilder


class PriorProbability(Initializer):
    """ Apply a prior probability to the weights.
    """

    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {
            'probability': self.probability
        }

    def __call__(self, shape, dtype=None):
        # set bias to -log((1 - p)/p) for foreground
        result = np.ones(shape, dtype=dtype) * -math.log((1 - self.probability) / self.probability)

        return result


def classify(number):
    inputs = Input(shape=(None, None, None, 256))
    outputs = inputs
    for i in range(4):
        outputs = Conv3D(
            filters=256, activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            kernel_size=3, strides=1, padding='same'
        )(outputs)

    outputs = Conv3D(
        filters=2 * 1,
        kernel_initializer=zeros(),
        bias_initializer=PriorProbability(probability=0.01),
        name='pyramid_classification',
        kernel_size=3, strides=1, padding='same'
    )(outputs)

    outputs = Reshape((-1, 2), name='pyramid_regression_reshape')(outputs)
    outputs = Activation(
        'sigmoid',
        name='pyramid_classification_sigmoid'
    )(outputs)

    return Model(inputs=inputs, outputs=outputs,
                 name='pyramid_classification_{}'.format(number))


def regress(number):
    inputs = Input(shape=(None, None, None, 256))
    outputs = inputs
    for i in range(4):
        outputs = Conv3D(
            filters=256, activation='relu',
            name='pyramid_regression_{}'.format(i),
            kernel_initializer=normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            kernel_size=3, strides=1, padding='same'
        )(outputs)

    outputs = Conv3D(
        filters=1 * 6,
        kernel_initializer=zeros(),
        bias_initializer=PriorProbability(probability=0.01),
        name='pyramid_regression',
        kernel_size=3, strides=1, padding='same'
    )(outputs)

    outputs = Reshape((-1, 6), name='pyramid_regression_reshape')(outputs)
    outputs = Activation(
        'sigmoid',
        name='pyramid_regression_sigmoid'
    )(outputs)

    return Model(inputs=inputs, outputs=outputs,
                 name='pyramid_regression_{}'.format(number))


class Anchors(Layer):
    """ Keras layer to regress boxes.
    """

    def __init__(self, img_sz, num_anchors, *args, **kwargs):
        self.img_sz = img_sz
        self.num_anchors = num_anchors
        super(Anchors, self).__init__(*args, **kwargs)

    def __generate_anchor_offsets(self, img_sz, stride):
        ratios = np.array([(1, 1, 1)])
        # scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
        scales = np.array([2 ** 0])
        num_anchors = len(ratios) * len(scales)

        ratios_tiled = np.tile(ratios, len(scales))
        ratios_tiled = ratios_tiled.reshape(num_anchors, len(ratios[0]))
        anchors = stride * np.tile(scales, (3, len(ratios))).T
        anchors = ratios_tiled * anchors
        anchors_offset_1 = (anchors / 2) - anchors
        anchors_offset_2 = anchors - (anchors / 2)
        anchors = np.concatenate((anchors_offset_1, anchors_offset_2), axis=1)
        block_sz = img_sz // stride
        anchors = np.tile(anchors, (block_sz * block_sz * block_sz, 1))
        anchors = np.reshape(anchors, (block_sz * block_sz * block_sz * num_anchors, 6))
        return anchors

    def call(self, inputs, **kwargs):
        stride = K.int_shape(inputs)[-2]
        batch = K.shape(inputs)[0]
        block_sz = self.img_sz // stride
        center = (K.arange(0, block_sz, dtype=K.floatx()) +
                  K.constant(0.5, dtype=K.floatx())) * stride
        # Weird bug
        Y, X, Z = tf.meshgrid(center, center, center)

        X = tf.reshape(tf.tile(tf.expand_dims(tf.reshape(X, [-1]), -1),
                               [1, self.num_anchors]), [-1])
        Y = tf.reshape(tf.tile(tf.expand_dims(tf.reshape(Y, [-1]), -1),
                               [1, self.num_anchors]), [-1])
        Z = tf.reshape(tf.tile(tf.expand_dims(tf.reshape(Z, [-1]), -1),
                               [1, self.num_anchors]), [-1])
        shifts = K.stack([X, Y, Z, X, Y, Z], axis=1)

        # Transform to (Block X * Block Y * Block Z), where 8 represents
        # the 8 bounding box coordinates, and 3 the xyz respectively
        shifts = tf.reshape(shifts, [block_sz * block_sz * block_sz * self.num_anchors, 6])
        anchors = self.__generate_anchor_offsets(self.img_sz, stride)
        anchors = anchors + shifts
        anchors = K.round(anchors)
        anchors = K.tile(K.expand_dims(anchors, axis=0), (batch, 1, 1))
        return anchors

    def compute_output_shape(self, input_shape):
        return (None, (input_shape[2] ** 3) * self.num_anchors, 6)


class RegressBoxes(Layer):
    """ Keras layer to regress boxes.
    """

    def call(self, inputs, **kwargs):
        boxes, deltas = inputs

        length  = boxes[:, :, 3] - boxes[:, :, 0]
        width  = boxes[:, :, 4] - boxes[:, :, 1]
        height = boxes[:, :, 5] - boxes[:, :, 2]

        x1 = boxes[:, :, 0] + deltas[:, :, 0] * length
        y1 = boxes[:, :, 1] + deltas[:, :, 1] * width
        z1 = boxes[:, :, 2] + deltas[:, :, 2] * height
        x2 = boxes[:, :, 3] + deltas[:, :, 3] * length
        y2 = boxes[:, :, 4] + deltas[:, :, 4] * width
        z2 = boxes[:, :, 5] + deltas[:, :, 5] * height

        x1 = tf.clip_by_value(boxes[:, :, 0], 0, 128)
        y1 = tf.clip_by_value(boxes[:, :, 1], 0, 128)
        z1 = tf.clip_by_value(boxes[:, :, 2], 0, 128)
        x2 = tf.clip_by_value(boxes[:, :, 3], 0, 128)
        y2 = tf.clip_by_value(boxes[:, :, 4], 0, 128)
        z2 = tf.clip_by_value(boxes[:, :, 5], 0, 128)

        return K.stack([x1, y1, z1, x2, y2, z2], axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class FilterDetection(Layer):
    """ Keras layer to regress boxes.
    """

    def __init__(self, score_threshold=0.05, max_detections=2000,
                 *args, **kwargs):
        self.score_threshold=0.05
        self.max_detections=2000
        super(FilterDetection, self).__init__(*args, **kwargs)

    def __filter_helper(self, boxes, classification):
        scores = K.max(classification, axis=1)
        labels = K.argmax(classification, axis=1)

        confidence_indices = tf.where(K.greater(scores, self.score_threshold))
        scores = tf.gather_nd(scores, confidence_indices)
        scores, top_indices = tf.nn.top_k(
            scores,
            k=K.minimum(self.max_detections, K.shape(scores)[0])
        )

        final_indices = K.gather(confidence_indices[:, 0], top_indices)
        boxes = tf.gather(boxes, final_indices)
        labels = tf.gather(labels, final_indices)

        pad_size = K.maximum(0, self.max_detections - K.shape(scores)[0])
        boxes = tf.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
        scores = tf.pad(scores, [[0, pad_size]], constant_values=-1)
        labels = tf.pad(labels, [[0, pad_size]], constant_values=-1)
        labels = K.cast(labels, 'int32')

        boxes.set_shape([self.max_detections, 6])
        scores.set_shape([self.max_detections])
        labels.set_shape([self.max_detections])

        return [boxes, scores, labels]

    def call(self, inputs, **kwargs):
        boxes, classification = inputs

        def __filter_helper(args):
            return self.__filter_helper(args[0], args[1])

        # call filter_detections on each batch
        return tf.map_fn(
            __filter_helper,
            elems=[boxes, classification],
            dtype=[K.floatx(), K.floatx(), 'int32'],
            parallel_iterations=32
        )

    def compute_output_shape(self, input_shape):
        return [
            (input_shape[0][0], self.max_detections, 6),
            (input_shape[1][0], self.max_detections),
            (input_shape[1][0], self.max_detections),
        ]


def build_retinanet():
    backbone = FPNBuilder.build()
    features = backbone.output
    classified = Concatenate(axis=1, name='classified_concat')([classify(i)(f) for i, f in enumerate(features)])
    regressed = Concatenate(axis=1, name='regressed_concat')([regress(i)(f) for i, f in enumerate(features)])

    anchors = [Anchors(128, 1)(f) for f in features]
    anchors = Concatenate(axis=1, name='anchors')(anchors)
    boxes = RegressBoxes(name='boxes')([anchors, regressed])
    output = FilterDetection()([boxes, classified])

    return keras.models.Model(inputs=backbone.input,
                              outputs=output,
                              name='basic_retinanet')
