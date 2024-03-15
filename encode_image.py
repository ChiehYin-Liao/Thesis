import bchlib
import glob
import os
from PIL import Image,ImageOps
import numpy as np
import tensorflow as tf
import tensorflow.contrib.image
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants

from tqdm import tqdm

import matplotlib.pyplot as plt

import splitfolders

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

import csv

BCH_POLYNOMIAL = 137
BCH_BITS = 5

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    # parser.add_argument('--image', type=str, default=None)
    # parser.add_argument('--images_dir', type=str, default=images_dir_path)
    # parser.add_argument('--save_dir', type=str, default=save_dir_path)
    parser.add_argument('--secret', type=str, default='Stega!!')
    parser.add_argument("--cuda", type=str, default=0)
    parser.add_argument('--split_folder_seed', type=int, default=1337)
    parser.add_argument('--is_test', action='store_true')
    args = parser.parse_args()  

    images_dir = './datasets/img_align_celeba_128x128/'
    save_dir = './encoded_image/' + args.model    
    # images_dir = '/scratch3/users/JIN/stylegan2/results/00031-generate-images-stylegan2-celeb_A_150k_identity_no_wrap'
    # save_dir = '/scratch3/users/JIN/StegaStamp/encoded_image/' + gan_model + '/' + args.model    

    if images_dir is not None:
        files_list = glob.glob(images_dir + '/*.png')
    else:
        print('Missing input image')
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)  

    sess = tf.InteractiveSession(graph=tf.Graph())

    model_path='/scratch3/users/JIN/StegaStamp/saved_models/' + args.model
    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], model_path)

    input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_secret = tf.get_default_graph().get_tensor_by_name(input_secret_name)
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['stegastamp'].name
    output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['residual'].name
    output_stegastamp = tf.get_default_graph().get_tensor_by_name(output_stegastamp_name)
    output_residual = tf.get_default_graph().get_tensor_by_name(output_residual_name)

    width = 128
    height = 128
    psnr = 0
    ssim = 0

    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    if len(args.secret) > 7:
        print('Error: Can only encode 56bits (7 characters) with ECC')
        return

    data = bytearray(args.secret + ' '*(7-len(args.secret)), 'utf-8')
    ecc = bch.encode(data) # encode error control code
    packet = data + ecc # add error control code to the original bit string

    packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(x) for x in packet_binary]
    secret.extend([0,0,0,0])

    hidden_dir = save_dir + '/' + args.model + '_hidden' + '/class1'

    if args.is_test :
        image_length = 6667
        # image_length = 1
    else:
        image_length = len(files_list)
    print(image_length)

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(hidden_dir):
            os.makedirs(hidden_dir)
        size = (width, height)
        for idx, filename in tqdm(enumerate(files_list)):
            if idx < image_length:
                image = Image.open(filename).convert("RGB")
                image = np.array(ImageOps.fit(image,size),dtype=np.float32)
                image /= 255.

                feed_dict = {input_secret:[secret],
                            input_image:[image]}

                hidden_img, residual = sess.run([output_stegastamp, output_residual],feed_dict=feed_dict)
                # print(hidden_img.shape)
                rescaled = (hidden_img[0] * 255).astype(np.uint8)
                raw_img = (image * 255).astype(np.uint8)
                psnr += peak_signal_noise_ratio(raw_img, rescaled)
                ssim += structural_similarity(raw_img, rescaled, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
                
                residual = residual[0]+.5

                residual = (residual * 255).astype(np.uint8)

                save_name = filename.split('/')[-1].split('.')[0]

                im = Image.fromarray(np.array(rescaled))
                im.save(save_dir + '/'+save_name+'_hidden.png')
                im.save(hidden_dir + '/'+save_name+'_hidden.png')

                im = Image.fromarray(np.squeeze(np.array(residual)))
                im.save(save_dir + '/'+save_name+'_residual.png')
            else:
                break  

    output_folder =  '/scratch3/users/JIN/StegaStamp/encoded_image/' + args.model + '_hidden'
    input_folder = save_dir + '/' + args.model + '_hidden'
    psnr = str(psnr/image_length)
    ssim = str(ssim/image_length)

    accuracy_dir = './accuracy_csv'
    if not os.path.exists(accuracy_dir):
        print(f"make dir: {accuracy_dir}")
        os.makedirs(accuracy_dir)
    accuracy_path = os.path.join(accuracy_dir, "accuracy_of_all.txt")
    
    with open(accuracy_path, 'a') as f:
        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        row = []
        row.append('PSNR')
        row.append(args.model)
        row.append(psnr)
        writer.writerow(row)
        row = []
        row.append('SSIM')
        row.append(args.model)
        row.append(ssim)
        writer.writerow(row)
    
    # Split with a ratio.
    # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
    splitfolders.ratio(input_folder, output=output_folder, seed=args.split_folder_seed, ratio=(.75, .25), group_prefix=None, move=False) # default values


if __name__=="__main__":
        main()