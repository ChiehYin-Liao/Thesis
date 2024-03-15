import numpy as np
import os
import lpips.lpips_tf as lpips_tf
import tensorflow as tf
import utils
from tensorflow import keras
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from stn import spatial_transformer_network as stn_transformer
# import tensorflow_addons as tfa

import logging
import sys
from logging.handlers import TimedRotatingFileHandler

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s")
LOG_FILE = "./log_of_running_exp/log_of_running_exp.log"


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler
def get_file_handler():
    file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight')
    file_handler.setFormatter(FORMATTER)
    return file_handler
def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG) # better to have too much log than not enough
    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler())
    # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = False
    return logger

test_model = 'celeb_A_200k_only_jpeg'

# my_logger = get_logger("logger")

class Attacker(Layer):
    def __init__(self):
        super(Attacker, self).__init__()
        self.attacker = Sequential([
            Conv2D(16, 11, activation=None, padding='same', kernel_initializer='he_normal'),
            LeakyReLU(alpha=0.3),
            Conv2D(3, 11, activation=None, padding='same', kernel_initializer='he_normal'),
            # Conv2D(3, 7, activation=None, padding='same', kernel_initializer='he_normal', dilation_rate=2),
            # LeakyReLU(alpha=0.3),
            # Conv2D(3, 5, activation=None, padding='same', kernel_initializer='he_normal'),
        ])

    def call(self, image):
            image = image - .5
            return self.attacker(image)


def transform_net(encoded_image, args):
    sh = tf.shape(encoded_image)

    # ramp_fn = lambda ramp : tf.minimum(tf.to_float(global_step) / ramp, 1.)


    # blur
    # f = utils.blur_kernel(probs=[.25,.25], N_blur=args.blur_attack_kernel_size,
    # f = utils.random_blur_kernel(probs=[.25,.25], N_blur=args.blur_attack_kernel_size,
    #                        sigrange_gauss=[args.blur_attack_sig,args.blur_attack_sig], sigrange_line=[.25,1.], wmin_line=3)
    kernel = utils.gaussian_kernel(size=args.blur_attack_kernel_size,sigma=args.blur_attack_sig)
    kernel = kernel[:, :, np.newaxis, np.newaxis] # height,width, channel_in, channel_out
    # x = tf.placeholder(tf.float32,shape=(None,200,300,3))    # (batch_size, height, width, channel)
    xr,xg,xb =tf.expand_dims(encoded_image[:,:,:,0],-1),tf.expand_dims(encoded_image[:,:,:,1],-1),tf.expand_dims(encoded_image[:,:,:,2],-1)

    xr_blur = tf.nn.conv2d(xr, kernel, strides=[1, 1, 1, 1], padding='SAME')
    xg_blur = tf.nn.conv2d(xg, kernel, strides=[1, 1, 1, 1], padding='SAME')
    xb_blur = tf.nn.conv2d(xb, kernel, strides=[1, 1, 1, 1], padding='SAME')
    # encoded_image = tf.concat([xr_blur,xg_blur,xb_blur],axis=3)
    encoded_image = tf.reverse(encoded_image, [2])

    encoded_image = tf.reshape(encoded_image, [-1,128,128,3])

    return encoded_image


def get_secret_acc(secret_true,secret_pred):
    with tf.variable_scope("acc"):
        secret_pred = tf.round(tf.sigmoid(secret_pred))
        correct_pred = tf.to_int64(tf.shape(secret_pred)[1]) - tf.count_nonzero(secret_pred - secret_true, axis=1)

        str_acc = 1.0 - tf.count_nonzero(correct_pred - tf.to_int64(tf.shape(secret_pred)[1])) / tf.size(correct_pred, out_type=tf.int64)

        bit_acc = tf.reduce_sum(correct_pred) / tf.size(secret_pred, out_type=tf.int64)
        return bit_acc, str_acc

def build_model(attacker,
                image_input,
                args,
                global_step):

    # my_logger.debug('no_wrap = ' + str(args.no_wrap))
    # my_logger.debug('no_jpeg = ' + str(args.no_jpeg))
    
    transformed_image = transform_net(image_input, args)
    attacked_image = attacker(image_input)

    image_mse = tf.keras.losses.MeanSquaredError()
    image_loss_op = image_mse(transformed_image, attacked_image)

    loss_op = image_loss_op

    summary_op = tf.summary.merge([
        # tf.summary.scalar('bit_acc', bit_acc, family='train'),
        tf.summary.scalar('loss', loss_op, family='train'),
        # tf.summary.scalar('image_loss', image_loss_op, family='train'),
        # tf.summary.scalar('lpip_loss', lpips_loss_op, family='train'),
        # tf.summary.scalar('G_loss', G_loss, family='train'),
        # tf.summary.scalar('secret_loss', secret_loss_op, family='train'),
        # tf.summary.scalar('dis_loss', D_loss, family='train'),
        # tf.summary.scalar('attacked_secret_loss', attacked_secret_loss_op, family='train'),

        # tf.summary.scalar('bit_attacked_acc', bit_attacked_acc, family='train_attacked'),
        # tf.summary.scalar('attacked_loss', attacked_loss_op, family='train_attacked'),
        # tf.summary.scalar('attacked_image_loss', attacked_image_loss_op, family='train_attacked'),
        # tf.summary.scalar('attacked_lpips_loss', attacked_lpips_loss_op, family='train_attacked'),
        # tf.summary.scalar('attacked_secret_loss', attacked_secret_loss_op, family='train_attacked'),
    ])

    image_summary_op = tf.summary.merge([
        image_to_summary(image_input, 'image_input', family='input'),
        image_to_summary(transformed_image, 'residual', family='transformed'),
        image_to_summary(attacked_image, 'attacked_image', family='attacked'),
        image_to_summary(attacked_image-transformed_image+.5, 'attacked_residual', family='attacked'),
    ])

    return loss_op, summary_op, image_summary_op

def image_to_summary(image, name, family='train'):
    image = tf.clip_by_value(image, 0, 1)
    image = tf.cast(image * 255, dtype=tf.uint8)
    summary = tf.summary.image(name,image,max_outputs=1,family=family)
    return summary

def prepare_deployment_graph(attacker, image_input, args):
    transformed_image = transform_net(image_input, args)
    attacked_image = attacker(image_input)
    residual = attacked_image - transformed_image + .5
    transformed_image = tf.clip_by_value(transformed_image, 0, 1)
    attacked_image = tf.clip_by_value(attacked_image, 0, 1)
    

    return attacked_image, transformed_image, residual
