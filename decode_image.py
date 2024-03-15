import bchlib
import glob
import os
from PIL import Image, ImageOps
import cv2

import numpy as np
import tensorflow as tf
import tensorflow.contrib.image
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants

from io import BytesIO
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

import logging
import sys
from logging.handlers import TimedRotatingFileHandler

from skimage.filters import gaussian
import csv
import image_transforms

BCH_POLYNOMIAL = 137
BCH_BITS = 5
FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
# parser.add_argument('--image', type=str, default=None)
# parser.add_argument('--images_dir', type=str, default=images_dir)
parser.add_argument('--secret_size', type=int, default=100)
parser.add_argument("--cuda", type=str, default=0)
parser.add_argument('--gan_model', type=str, default=None)
parser.add_argument('--gan_image_path', type=str, default=None)
parser.add_argument('--embeded_string', type=str, default='0101001101110100011001010110011101100001001000010010000110001000101010011111101101001110010000000000')
args = parser.parse_args()


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler
def get_file_handler(LOG_FILE):
    file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight')
    file_handler.setFormatter(FORMATTER)
    return file_handler
def get_logger(logger_name, LOG_FILE):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG) # better to have too much log than not enough
    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler(LOG_FILE))
    # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = False
    return logger

def addJPEGcompression(image, qf):
#     qf = random.randrange(10, 100)
    outputIoStream = BytesIO()
    image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
    outputIoStream.seek(0)
    return Image.open(outputIoStream)

def add_gaussian_noise(image, std):
    # image must be scaled in [0, 1]
    if std is 0:
        return image
    else:
        noise = np.random.normal(loc=0.0, scale=std, size=image.shape)
        noise_img = image + noise
        noise_img = np.clip(noise_img, 0.0, 1.0)
        # print(type(noise_img), type(image), type(noise))        
        return noise_img

def crop_center(img, cropx, cropy):
    y, x, _ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy, startx:startx+cropx]

def calculate_accuracy(noise, qf):
    total_bitwise_accuracy = 0
    face_image_path = './datasets/img_align_celeba_128x128/000011.png'
    len_file_list = len(files_list)
    for idx, filename in tqdm(enumerate(files_list)):
        if noise is 'faceswap':
            try:
                image = image_transforms.swap_faces(face_image_path, filename)
            except:
                len_file_list -= 1
                continue
            image = np.array(image, dtype=np.float32)
            image /= 255.
        else:    
            image = Image.open(filename).convert("RGB")
                # plt.imshow(image)
                # plt.show()
            
            # add distortion before decode
            if noise is 'identity':
                image = np.array(ImageOps.fit(image,(128, 128)),dtype=np.float32)
                image /= 255.

            elif noise is 'Gaussian_noise':
                image = np.array(ImageOps.fit(image,(128, 128)),dtype=np.float32)
                image /= 255.
                image = add_gaussian_noise(image, qf)

            elif noise is 'Gaussian_blur':
                image = np.array(ImageOps.fit(image,(128, 128)),dtype=np.float32)
                image = gaussian(image, sigma=qf, multichannel=True)
                image /= 255.

            elif noise is 'jpeg':
                image = addJPEGcompression(image, qf)
                image = np.array(ImageOps.fit(image,(128, 128)),dtype=np.float32)
                image /= 255.  # shape = (128, 128, 3)

            elif noise is 'CenterCrop':
                image = np.array(ImageOps.fit(image,(128, 128)),dtype=np.float32)
                image = crop_center(image, qf, qf)
                # image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
                image /= 255. 
                pad_length = (128 - qf)//2
                image = np.pad(image, ((pad_length,pad_length), (pad_length, pad_length), (0, 0)))

            elif noise is 'flip':
                image = np.array(ImageOps.fit(image,(128, 128)),dtype=np.float32)
                image /= 255.
                image = np.fliplr(image)
        
        feed_dict = {input_image:[image]}
        secret = sess.run([output_secret],feed_dict=feed_dict)[0][0]
        packet_binary = "".join([str(int(bit)) for bit in secret[:100]])
        bitwise_accuracy = sum ( args.embeded_string[i] == packet_binary[i] for i in range(100) ) / 100
        total_bitwise_accuracy += bitwise_accuracy
        # if idx%1000 is 0:
        if idx%1000 is 0:
            my_logger.debug('idx = '+str(idx)+', qf = '+str(qf)+' , mean_bitwise_accuracy = '+str(total_bitwise_accuracy/(idx+1)))
        # my_logger.debug('idx = '+str(idx)+', qf = '+str(qf)+' , mean_bitwise_accuracy = '+str(bitwise_accuracy))
        # my_logger.debug(str(bitwise_accuracy))
    # my_logger.info('idx = '+str(len_file_list)+', qf = '+str(qf)+' , mean_bitwise_accuracy = '+str(total_bitwise_accuracy/len_file_list))
    return total_bitwise_accuracy/len_file_list


def get_distortion_value(noise, distortion_iteration):
    if noise is 'Gaussian_noise':
        return (round((distortion_iteration) * 0.055, 2))
    elif noise is 'Gaussian_blur':
        # v = round((distortion_iteration) * 2 - 1, 2)
        # return 0 if v < 0 else v
        # return distortion_iteration
        return round(distortion_iteration*2, 2)
    elif noise is 'jpeg':
        return (100 - (distortion_iteration)*10)
    elif noise is 'CenterCrop':
        return (128-round((distortion_iteration) * 8, 2))

def get_distortion_accuracy(noise):
    distortion_bitwise_accuracy = []
    # distortion_bitwise_accuracy.append(calculate_accuracy('identity', 0))
    for distortion_iteration in range(10):
        qf = get_distortion_value(noise, distortion_iteration)
        print(qf)
        distortion_bitwise_accuracy.append(calculate_accuracy(noise, qf))
    return distortion_bitwise_accuracy  

def draw_plot(noise, plt_color, plt_title, plt_xlabel, plt_savefig):    
    distortion_value = []
    bitwise_accuracy = []
    if noise == 'faceswap' or noise == 'flip' or noise == 'identity':
        bitwise_accuracy.append(calculate_accuracy(noise, 0))
    else:
        bitwise_accuracy = get_distortion_accuracy(noise)    
        for distortion_iteration in range(10):
            distortion_value.append(get_distortion_value(noise, distortion_iteration))
    my_logger.info(noise)
    my_logger.info(bitwise_accuracy)
    my_logger.info(distortion_value)

    with open(os.path.join('./accuracy_csv/', "accuracy_of_all.txt"), 'a') as f:
        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        row = []
        row.append(noise)
        if args.gan_model is None:
            row.append(args.model)
        else:
            row.append(args.gan_model)
        row += bitwise_accuracy
        # row.append('\n')
        writer.writerow(row)

    # plt.plot(distortion_value, bitwise_accuracy, color=plt_color, marker='o')
    # plt.title(plt_title, fontsize=14)
    # plt.xlabel(plt_xlabel, fontsize=14)
    # plt.ylabel('Bitwise_accuracy', fontsize=14)
    # plt.grid(True)
    # plt.show()
    # plt.savefig(plt_savefig)
    # plt.clf()


if args.gan_model is None:
    # images_dir = '/scratch3/users/JIN/StegaStamp/encoded_image/' + args.model +'_hidden/val/class1'
    images_dir = '/scratch3/users/JIN/StegaStamp/encoded_image/' + args.model +'_hidden/train/class1'
else:
    images_dir = '/scratch3/users/JIN/StegaStamp/encoded_image/' + args.gan_model

if images_dir is not None:
    if args.gan_model is None:
        files_list = glob.glob(images_dir + '/*_hidden.png')
    else:
        files_list = glob.glob(images_dir + '/*.png')

else:
    print('Missing input image')

if args.gan_model is None:
    LOG_FILE = "/scratch3/users/JIN/StegaStamp/decode_image_accuaracy_log/" + args.model + ".log"
else:
    # if not os.path.exists("/scratch3/users/JIN/StegaStamp/decode_image_accuaracy_log/" + args.gan_model):
    #     os.makedirs("/scratch3/users/JIN/StegaStamp/decode_image_accuaracy_log/" + args.gan_model)  
    LOG_FILE = "/scratch3/users/JIN/StegaStamp/decode_image_accuaracy_log/" + args.gan_model + ".log"

my_logger = get_logger("logger", LOG_FILE)
if args.gan_model is None:
    my_logger.info(args.model)
else:
    my_logger.info(args.gan_model)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

sess = tf.InteractiveSession(graph=tf.Graph())

model_path = '/scratch3/users/JIN/StegaStamp/saved_models/' + args.model
model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], model_path)

input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

output_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['decoded'].name
output_secret = tf.get_default_graph().get_tensor_by_name(output_secret_name)

bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

# Stega!!'s bianary string'
# embeded_string = '0101001101110100011001010110011101100001001000010010000110001000101010011111101101001110010000000000'
if args.gan_model is None:
    save_fig_path = '/scratch3/users/JIN/StegaStamp/accuracy_figure/' + args.model
else:
    save_fig_path = '/scratch3/users/JIN/StegaStamp/accuracy_figure/' + args.gan_model

if not os.path.exists(save_fig_path):
    os.makedirs(save_fig_path) 

if args.gan_model is None:
    # draw_plot(noise='identity', plt_color='blue', plt_title='identity-'+args.model, plt_xlabel='identity', plt_savefig=save_fig_path+'/identity-' + time.strftime("%Y-%m-%d-%H") +'.pdf')
    draw_plot(noise='jpeg', plt_color='blue', plt_title='JPEG Compression-'+args.model, plt_xlabel='JPEG quality', plt_savefig=save_fig_path+'/JPEG-Compression-' + time.strftime("%Y-%m-%d-%H") +'.pdf')
    draw_plot(noise='Gaussian_noise', plt_color='blue', plt_title='Gaussian Noise-'+args.model, plt_xlabel='Noise std', plt_savefig=save_fig_path+'/Gaussian_noise-' + time.strftime("%Y-%m-%d-%H") +'.pdf')
    draw_plot(noise='Gaussian_blur', plt_color='blue', plt_title='Blurring-'+args.model, plt_xlabel='Kernel size', plt_savefig=save_fig_path+'/Gaussian_blur-' + time.strftime("%Y-%m-%d-%H") +'.pdf')
    draw_plot(noise='CenterCrop', plt_color='blue', plt_title='Center Cropping-'+args.model, plt_xlabel='Crop size', plt_savefig=save_fig_path+'/CenterCrop-' + time.strftime("%Y-%m-%d-%H") +'.pdf')
    draw_plot(noise='flip', plt_color='blue', plt_title='flip-'+args.model, plt_xlabel='flip', plt_savefig=save_fig_path+'/flip-' + time.strftime("%Y-%m-%d-%H") +'.pdf')
    # draw_plot(noise='faceswap', plt_color='blue', plt_title='faceswap-'+args.model, plt_xlabel='faceswap', plt_savefig=save_fig_path+'/faceswap-' + time.strftime("%Y-%m-%d-%H") +'.pdf')
else:
    # draw_plot(noise='identity', plt_color='blue', plt_title='identity-'+args.gan_model, plt_xlabel='identity', plt_savefig=save_fig_path+'/identity-' + time.strftime("%Y-%m-%d-%H") +'.pdf')
    draw_plot(noise='jpeg', plt_color='blue', plt_title='JPEG Compression-'+args.gan_model, plt_xlabel='JPEG quality', plt_savefig=save_fig_path+'/JPEG-Compression-' + time.strftime("%Y-%m-%d-%H") +'.pdf')
    draw_plot(noise='Gaussian_noise', plt_color='blue', plt_title='Gaussian Noise-'+args.gan_model, plt_xlabel='Noise std', plt_savefig=save_fig_path+'/Gaussian_noise-' + time.strftime("%Y-%m-%d-%H") +'.pdf')
    draw_plot(noise='Gaussian_blur', plt_color='blue', plt_title='Blurring-'+args.gan_model, plt_xlabel='Kernel size', plt_savefig=save_fig_path+'/Gaussian_blur-' + time.strftime("%Y-%m-%d-%H") +'.pdf')
    draw_plot(noise='CenterCrop', plt_color='blue', plt_title='Center Cropping-'+args.gan_model, plt_xlabel='Crop size', plt_savefig=save_fig_path+'/CenterCrop-' + time.strftime("%Y-%m-%d-%H") +'.pdf')
    draw_plot(noise='flip', plt_color='blue', plt_title='flip-'+args.gan_model, plt_xlabel='flip', plt_savefig=save_fig_path+'/flip-' + time.strftime("%Y-%m-%d-%H") +'.pdf')
    # draw_plot(noise='faceswap', plt_color='blue', plt_title='faceswap-'+args.gan_model, plt_xlabel='faceswap', plt_savefig=save_fig_path+'/faceswap-' + time.strftime("%Y-%m-%d-%H") +'.pdf')
