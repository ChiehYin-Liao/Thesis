import numpy as np
import os
import lpips.lpips_tf as lpips_tf
import tensorflow as tf
import utils
from tensorflow import keras
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from stn import spatial_transformer_network as stn_transformer

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

class StegaStampEncoder(Layer):
    def __init__(self, height, width):
        super(StegaStampEncoder, self).__init__()
        self.secret_dense = Dense(768, activation='relu', kernel_initializer='he_normal')
        # 768 = 16 * 16 * 3
        self.conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv2 = Conv2D(32, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')
        self.conv3 = Conv2D(64, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')
        self.conv4 = Conv2D(128, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')
        self.conv5 = Conv2D(256, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')
        self.up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up8 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.residual = Conv2D(3, 1, activation=None, padding='same', kernel_initializer='he_normal')

    def call(self, inputs):
        secret, image = inputs
        secret = secret - .5
        image = image - .5

        secret = self.secret_dense(secret)
        # secret = Reshape((50, 50, 3))(secret)
        secret = Reshape((16, 16, 3))(secret)
        secret_enlarged = UpSampling2D(size=(8,8))(secret)

        # print(secret_enlarged.shape)

        inputs = concatenate([secret_enlarged, image], axis=-1)
        print(inputs.shape)
        conv1 = self.conv1(inputs)
        print(conv1.shape)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        up6 = self.up6(UpSampling2D(size=(2,2))(conv5))
        merge6 = concatenate([conv4,up6], axis=3)
        conv6 = self.conv6(merge6)
        up7 = self.up7(UpSampling2D(size=(2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis=3)
        conv7 = self.conv7(merge7)
        up8 = self.up8(UpSampling2D(size=(2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis=3)
        conv8 = self.conv8(merge8)
        up9 = self.up9(UpSampling2D(size=(2,2))(conv8))
        merge9 = concatenate([conv1,up9,inputs], axis=3)
        conv9 = self.conv9(merge9)
        conva = self.conv9(merge9)
        conv10 = self.conv10(conv9)
        residual = self.residual(conv9)
        return tf.sigmoid(residual)
        # return residual

class StegaStampDecoder(Layer):
    def __init__(self, secret_size, height, width):
        super(StegaStampDecoder, self).__init__()
        self.height = height
        self.width = width
        self.stn_params = Sequential([
            Conv2D(32, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(64, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(128, (3, 3), strides=2, activation='relu', padding='same'),
            Flatten(),
            Dense(128, activation='relu')
        ])
        initial = np.array([[1., 0, 0], [0, 1., 0]])
        initial = initial.astype('float32').flatten()

        self.W_fc1 = tf.Variable(tf.zeros([128, 6]), name='W_fc1')
        self.b_fc1 = tf.Variable(initial_value=initial, name='b_fc1')

        self.decoder = Sequential([
            Conv2D(32, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(128, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(128, (3, 3), strides=2, activation='relu', padding='same'),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(secret_size)
        ])

    def call(self, image):
        image = image - .5
        # stn_params = self.stn_params(image)
        # x = tf.matmul(stn_params, self.W_fc1) + self.b_fc1
        # transformed_image = stn_transformer(image, x, [self.height, self.width, 3])
        # return self.decoder(transformed_image)
        return self.decoder(image)

class Discriminator(Layer):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = Sequential([
            Conv2D(8, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(16, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(32, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(64, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(1, (3, 3), activation=None, padding='same')
        ])

    def call(self, image):
            x = image - .5
            # x = image
            x = self.model(x)
            output = tf.reduce_mean(x)
            return output, x

class Attacker(Layer):
    def __init__(self):
        super(Attacker, self).__init__()
        self.attacker = Sequential([
            Conv2D(16, 7, activation=None, padding='same', kernel_initializer='he_normal'),
            LeakyReLU(alpha=0.3),
            Conv2D(3, 3, activation=None, padding='same', kernel_initializer='he_normal'),
            # Conv2D(3, 7, activation=None, padding='same', kernel_initializer='he_normal', dilation_rate=2),
            # LeakyReLU(alpha=0.3),
            # Conv2D(3, 5, activation=None, padding='same', kernel_initializer='ones'),
        ])

    def call(self, image):
            image = image - .5
            return self.attacker(image)


def transform_net(encoded_image, args, global_step):
    sh = tf.shape(encoded_image)

    ramp_fn = lambda ramp : tf.minimum(tf.to_float(global_step) / ramp, 1.)

    rnd_bri = ramp_fn(args.rnd_bri_ramp) * args.rnd_bri
    rnd_hue = ramp_fn(args.rnd_hue_ramp) * args.rnd_hue
    rnd_brightness = utils.get_rnd_brightness_tf(rnd_bri, rnd_hue, args.batch_size)

    jpeg_quality = 100. - tf.random.uniform([]) * ramp_fn(args.jpeg_quality_ramp) * (100.-args.jpeg_quality)
    jpeg_factor = tf.cond(tf.less(jpeg_quality, 50), lambda: 5000. / jpeg_quality, lambda: 200. - jpeg_quality * 2) / 100. + .0001

    rnd_noise = tf.random.uniform([]) * ramp_fn(args.rnd_noise_ramp) * args.rnd_noise

    contrast_low = 1. - (1. - args.contrast_low) * ramp_fn(args.contrast_ramp)
    contrast_high = 1. + (args.contrast_high - 1.) * ramp_fn(args.contrast_ramp)
    contrast_params = [contrast_low, contrast_high]

    rnd_sat = tf.random.uniform([]) * ramp_fn(args.rnd_sat_ramp) * args.rnd_sat

    # blur
    f = utils.random_blur_kernel(probs=[.25,.25], N_blur=args.blur_attack_kernel_size,
    # f = utils.random_blur_kernel(probs=[.25,.25], N_blur=17,
                           sigrange_gauss=[1.,3.], sigrange_line=[.25,1.], wmin_line=3)
    encoded_image = tf.nn.conv2d(encoded_image, f, [1,1,1,1], padding='SAME')

    noise = tf.random_normal(shape=tf.shape(encoded_image), mean=0.0, stddev=rnd_noise, dtype=tf.float32)
    # encoded_image = encoded_image + noise
    # encoded_image = tf.clip_by_value(encoded_image, 0, 1)

    contrast_scale = tf.random_uniform(shape=[tf.shape(encoded_image)[0]], minval=contrast_params[0], maxval=contrast_params[1])
    contrast_scale = tf.reshape(contrast_scale, shape=[tf.shape(encoded_image)[0],1,1,1])

    # encoded_image = encoded_image * contrast_scale
    # encoded_image = encoded_image + rnd_brightness
    # encoded_image = tf.clip_by_value(encoded_image, 0, 1)


    # encoded_image_lum = tf.expand_dims(tf.reduce_sum(encoded_image * tf.constant([.3,.6,.1]), axis=3), 3)
    # encoded_image = (1 - rnd_sat) * encoded_image + rnd_sat * encoded_image_lum

    # add the stylegan2 mirror distortion
    # print(tf.shape(encoded_image))
    # print(encoded_image.get_shape())

    # encoded_image = tf.where(tf.random_uniform([tf.shape(encoded_image)[0]]) < 0.5, encoded_image, tf.reverse(encoded_image, [2]))
    

    encoded_image = tf.reshape(encoded_image, [-1,128,128,3])
    # print(tf.shape(encoded_image))
    # print(encoded_image.get_shape())
    # if not args.no_jpeg:
        # my_logger.debug('is JPEG!!')
        # encoded_image = utils.jpeg_compress_decompress(encoded_image, rounding=utils.round_only_at_0, factor=jpeg_factor, downsample_c=True)

    summaries = [tf.summary.scalar('transformer/rnd_bri', rnd_bri),
                 tf.summary.scalar('transformer/rnd_sat', rnd_sat),
                 tf.summary.scalar('transformer/rnd_hue', rnd_hue),
                 tf.summary.scalar('transformer/rnd_noise', rnd_noise),
                 tf.summary.scalar('transformer/contrast_low', contrast_low),
                 tf.summary.scalar('transformer/contrast_high', contrast_high),
                 tf.summary.scalar('transformer/jpeg_quality', jpeg_quality)]

    return encoded_image, summaries


def get_secret_acc(secret_true,secret_pred):
    with tf.variable_scope("acc"):
        secret_pred = tf.round(tf.sigmoid(secret_pred))
        correct_pred = tf.to_int64(tf.shape(secret_pred)[1]) - tf.count_nonzero(secret_pred - secret_true, axis=1)

        str_acc = 1.0 - tf.count_nonzero(correct_pred - tf.to_int64(tf.shape(secret_pred)[1])) / tf.size(correct_pred, out_type=tf.int64)

        bit_acc = tf.reduce_sum(correct_pred) / tf.size(secret_pred, out_type=tf.int64)
        return bit_acc, str_acc

def build_model(encoder,
                decoder,
                discriminator,
                attacker,
                secret_input,
                image_input,
                l2_edge_gain,
                borders,
                secret_size,
                M,
                loss_scales,
                yuv_scales,
                args,
                global_step):

    # my_logger.debug('no_wrap = ' + str(args.no_wrap))
    # my_logger.debug('no_jpeg = ' + str(args.no_jpeg))


    if not args.no_wrap:
        # my_logger.debug('is warp!!')

        input_warped = tf.contrib.image.transform(image_input, M[:,1,:], interpolation='BILINEAR')
        mask_warped = tf.contrib.image.transform(tf.ones_like(input_warped), M[:,1,:], interpolation='BILINEAR')
        input_warped += (1-mask_warped) * image_input

        residual_warped = encoder((secret_input, input_warped))
        encoded_warped = residual_warped + input_warped
        residual = tf.contrib.image.transform(residual_warped, M[:,0,:], interpolation='BILINEAR')

        if borders == 'no_edge':
            encoded_image = image_input + residual
        elif borders == 'black':
            encoded_image = residual_warped + input_warped
            encoded_image = tf.contrib.image.transform(encoded_image, M[:,0,:], interpolation='BILINEAR')
            input_unwarped = tf.contrib.image.transform(input_warped, M[:,0,:], interpolation='BILINEAR')
        elif borders.startswith('random'):
            mask = tf.contrib.image.transform(tf.ones_like(residual), M[:,0,:], interpolation='BILINEAR')
            encoded_image = residual_warped + input_warped
            encoded_image = tf.contrib.image.transform(encoded_image, M[:,0,:], interpolation='BILINEAR')
            input_unwarped = tf.contrib.image.transform(input_warped, M[:,0,:], interpolation='BILINEAR')
            ch = 3 if borders.endswith('rgb') else 1
            encoded_image += (1-mask) * tf.ones_like(residual) * tf.random.uniform([ch])
        elif borders == 'white':
            mask = tf.contrib.image.transform(tf.ones_like(residual), M[:,0,:], interpolation='BILINEAR')
            encoded_image = residual_warped + input_warped
            encoded_image = tf.contrib.image.transform(encoded_image, M[:,0,:], interpolation='BILINEAR')
            input_unwarped = tf.contrib.image.transform(input_warped, M[:,0,:], interpolation='BILINEAR')
            encoded_image += (1-mask) * tf.ones_like(residual)
        elif borders == 'image':
            mask = tf.contrib.image.transform(tf.ones_like(residual), M[:,0,:], interpolation='BILINEAR')
            encoded_image = residual_warped + input_warped
            encoded_image = tf.contrib.image.transform(encoded_image, M[:,0,:], interpolation='BILINEAR')
            encoded_image += (1-mask) * tf.manip.roll(image_input, shift=1, axis=0)

        if borders == 'no_edge':
            D_output_real, _ = discriminator(image_input)
            D_output_fake, D_heatmap = discriminator(encoded_image)
        else:
            D_output_real, _ = discriminator(input_warped)
            D_output_fake, D_heatmap = discriminator(encoded_warped)

    else:
        # my_logger.debug('no warp!!')
        encoded_image = encoder((secret_input, image_input))
        # encoded_image = image_input + residual
        residual =  encoded_image - image_input
        # encoded_image = tf.math.sigmoid(encoded_image)
        D_output_real, _ = discriminator(image_input)
        D_output_fake, D_heatmap = discriminator(encoded_image)

    decoded_secret = decoder(encoded_image)
    if not args.no_blur_attack:
        transformed_image, transform_summaries = transform_net(encoded_image, args, global_step)
        attacked_image = attacker(transformed_image)
    else:
        attacked_image = attacker(encoded_image)
    # transformed_image, transform_summaries = transform_net(encoded_image, args, global_step)
    # attacked_image = attacker(transformed_image)
    # print(encoded_image.shape)
    # print(attacked_image.shape)

    decoded_attacked_secret = decoder(attacked_image)
    

    bit_acc, str_acc = get_secret_acc(secret_input, decoded_secret)
    bit_attacked_acc, str_attacked_acc = get_secret_acc(secret_input, decoded_attacked_secret)

    lpips_loss_op = tf.reduce_mean(lpips_tf.lpips(image_input, encoded_image))
    attacked_lpips_loss_op = tf.reduce_mean(lpips_tf.lpips(encoded_image, attacked_image))
    secret_loss_op = tf.losses.sigmoid_cross_entropy(secret_input, decoded_secret)
    attacked_secret_loss_op = tf.losses.sigmoid_cross_entropy(secret_input, decoded_attacked_secret)

    # size = (int(image_input.shape[1]),int(image_input.shape[2]))
    # gain = 10
    # falloff_speed = 4 # Cos dropoff that reaches 0 at distance 1/x into image
    # falloff_im = np.ones(size)
    # for i in range(int(falloff_im.shape[0]/falloff_speed)):
    #     falloff_im[-i,:] *= (np.cos(4*np.pi*i/size[0]+np.pi)+1)/2
    #     falloff_im[i,:] *= (np.cos(4*np.pi*i/size[0]+np.pi)+1)/2
    # for j in range(int(falloff_im.shape[1]/falloff_speed)):
    #     falloff_im[:,-j] *= (np.cos(4*np.pi*j/size[0]+np.pi)+1)/2
    #     falloff_im[:,j] *= (np.cos(4*np.pi*j/size[0]+np.pi)+1)/2
    # falloff_im = 1-falloff_im
    # falloff_im = tf.convert_to_tensor(falloff_im, dtype=tf.float32)
    # falloff_im *= l2_edge_gain

    # encoded_image_yuv = tf.image.rgb_to_yuv(encoded_image)
    # image_input_yuv = tf.image.rgb_to_yuv(image_input)

    # im_diff = encoded_image-image_input
    # im_diff += im_diff * tf.expand_dims(falloff_im, axis=[-1])
    # image_loss_op = tf.reduce_mean(tf.square(im_diff))
    image_mse = tf.keras.losses.MeanSquaredError()
    image_loss_op = image_mse(image_input, encoded_image)
    # image_loss_op = tf.tensordot(yuv_loss_op, yuv_scales[0:3], axes=1)
    # image_loss_op = tf.keras.losses.MeanSquaredError(image_input, encoded_image)

    # attacked_encoded_image_yuv = tf.image.rgb_to_yuv(attacked_image)
    # attacked_im_diff = attacked_image-encoded_image
    # attacked_im_diff += attacked_im_diff * tf.expand_dims(falloff_im, axis=[-1])
    # attacked_image_loss_op = tf.reduce_mean(tf.square(attacked_im_diff))
    attacked_image_mse = tf.keras.losses.MeanSquaredError()
    attacked_image_loss_op = attacked_image_mse(encoded_image, attacked_image)
    # attacked_image_loss_op = attacked_image_mse(transformed_image, attacked_image)
    # attacked_image_loss_op = tf.tensordot(attacked_yuv_loss_op, yuv_scales[3:6], axes=1)


    D_loss = D_output_real - D_output_fake
    G_loss = D_output_fake

    
    # attacked_loss_op = loss_scales[4]*attacked_image_loss_op - loss_scales[6]*attacked_secret_loss_op
    attacked_loss_op = -loss_scales[6]*attacked_secret_loss_op
    loss_op = loss_scales[0]*image_loss_op + loss_scales[2]*secret_loss_op
    # loss_op = loss_scales[0]*image_loss_op + loss_scales[1]*lpips_loss_op + loss_scales[2]*secret_loss_op
    

    if not args.no_attack:
        loss_op += loss_scales[7]*attacked_secret_loss_op
    if not args.no_lpips:
        loss_op += loss_scales[1]*lpips_loss_op
    if not args.no_adv_lpips:
        attacked_loss_op += loss_scales[5]*attacked_lpips_loss_op
    if not args.no_adv_l2:
        attacked_loss_op += loss_scales[4]*attacked_image_loss_op
    if not args.no_gan:
        loss_op += loss_scales[3]*G_loss

    summary_op = tf.summary.merge([
        tf.summary.scalar('bit_acc', bit_acc, family='train'),
        tf.summary.scalar('loss', loss_op, family='train'),
        tf.summary.scalar('image_loss', image_loss_op, family='train'),
        tf.summary.scalar('lpip_loss', lpips_loss_op, family='train'),
        tf.summary.scalar('G_loss', G_loss, family='train'),
        tf.summary.scalar('secret_loss', secret_loss_op, family='train'),
        tf.summary.scalar('dis_loss', D_loss, family='train'),
        tf.summary.scalar('attacked_secret_loss', attacked_secret_loss_op, family='train'),

        tf.summary.scalar('bit_attacked_acc', bit_attacked_acc, family='train_attacked'),
        tf.summary.scalar('attacked_loss', attacked_loss_op, family='train_attacked'),
        tf.summary.scalar('attacked_image_loss', attacked_image_loss_op, family='train_attacked'),
        tf.summary.scalar('attacked_lpips_loss', attacked_lpips_loss_op, family='train_attacked'),
        tf.summary.scalar('attacked_secret_loss', attacked_secret_loss_op, family='train_attacked'),

        # tf.summary.scalar('Y_loss', yuv_loss_op[0], family='color_loss'),
        # tf.summary.scalar('U_loss', yuv_loss_op[1], family='color_loss'),
        # tf.summary.scalar('V_loss', yuv_loss_op[2], family='color_loss'),

        # tf.summary.scalar('Y_loss', attacked_yuv_loss_op[0], family='color_loss_attacked'),
        # tf.summary.scalar('U_loss', attacked_yuv_loss_op[1], family='color_loss_attacked'),
        # tf.summary.scalar('V_loss', attacked_yuv_loss_op[2], family='color_loss_attacked'),
    # ] + transform_summaries)
    ])

    image_summary_op = tf.summary.merge([
        image_to_summary(image_input, 'image_input', family='input'),
        # image_to_summary(input_warped, 'image_warped', family='input'),
        # image_to_summary(encoded_warped, 'encoded_warped', family='encoded'),
        image_to_summary(residual+.5, 'residual', family='encoded'),
        # image_to_summary(residual_warped+.5, 'residual', family='encoded'),
        image_to_summary(encoded_image, 'encoded_image', family='encoded'),
        image_to_summary(attacked_image, 'attacked_image', family='transformed'),
        image_to_summary(attacked_image-encoded_image+.5, 'attacked_residual', family='transformed'),
        image_to_summary(D_heatmap, 'discriminator', family='losses'),
    ])

    return loss_op, attacked_loss_op, secret_loss_op, D_loss, summary_op, image_summary_op, bit_acc

def image_to_summary(image, name, family='train'):
    image = tf.clip_by_value(image, 0, 1)
    image = tf.cast(image * 255, dtype=tf.uint8)
    summary = tf.summary.image(name,image,max_outputs=1,family=family)
    return summary

def prepare_deployment_hiding_graph(encoder, secret_input, image_input):

    encoded_image = encoder((secret_input, image_input))
    residual = encoded_image - image_input
    encoded_image = tf.clip_by_value(encoded_image, 0, 1)

    return encoded_image, residual

def prepare_deployment_reveal_graph(decoder, image_input):
    decoded_secret = decoded_secret = decoder(image_input)

    return tf.round(tf.sigmoid(decoded_secret))
