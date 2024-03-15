import glob
import os
from PIL import Image,ImageOps
import numpy as np
import random
import tensorflow as tf
import utils
import models_AF
from os.path import join


TRAIN_PATH = './datasets/img_align_celeba_128x128/'
# TRAIN_PATH = './datasets/ffhq_128x128/'
LOGS_Path = './logs/'
CHECKPOINTS_PATH = './checkpoints/'
SAVED_MODELS = './saved_models'

if not os.path.exists(CHECKPOINTS_PATH):
    os.makedirs(CHECKPOINTS_PATH)

def get_img_batch(files_list,
                  secret_size,
                  batch_size=4,
                  size=(128,128)):

    batch_cover = []
    batch_secret = []

    for i in range(batch_size):
        img_cover_path = random.choice(files_list)
        try:
            img_cover = Image.open(img_cover_path).convert("RGB")
            img_cover = ImageOps.fit(img_cover, size)
            img_cover = np.array(img_cover, dtype=np.float32) / 255.
        except:
            img_cover = np.zeros((size[0],size[1],3), dtype=np.float32)
        batch_cover.append(img_cover)

        secret = np.random.binomial(1, .5, secret_size)
        batch_secret.append(secret)

    batch_cover, batch_secret = np.array(batch_cover), np.array(batch_secret)
    return batch_cover, batch_secret

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str)
    parser.add_argument('--secret_size', type=int, default=20)
    parser.add_argument('--num_steps', type=int, default=140000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--adv_batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=.0001)
    parser.add_argument('--adv_lr', type=float, default=.0001)
    parser.add_argument('--l2_loss_scale', type=float, default=1.5)
    parser.add_argument('--adv_l2_loss_scale', type=float, default=1.5)
    parser.add_argument('--l2_loss_ramp', type=int, default=20000)
    parser.add_argument('--adv_l2_loss_ramp', type=int, default=20000)
    parser.add_argument('--l2_edge_gain', type=float, default=10)
    parser.add_argument('--l2_edge_ramp', type=int, default=10000)
    parser.add_argument('--l2_edge_delay', type=int, default=40000)
    parser.add_argument('--lpips_loss_scale', type=float, default=1)
    parser.add_argument('--adv_lpips_loss_scale', type=float, default=1)
    parser.add_argument('--lpips_loss_ramp', type=int, default=20000)
    parser.add_argument('--adv_lpips_loss_ramp', type=int, default=20000)
    parser.add_argument('--secret_loss_scale', type=float, default=1)
    parser.add_argument('--adv_secret_loss_scale', type=float, default=1)
    parser.add_argument('--adv_W_secret_loss_scale', type=float, default=1)
    parser.add_argument('--secret_loss_ramp', type=int, default=1)
    parser.add_argument('--adv_secret_loss_ramp', type=int, default=1)
    parser.add_argument('--adv_W_secret_loss_ramp', type=int, default=1)
    parser.add_argument('--G_loss_scale', type=float, default=1)
    parser.add_argument('--G_loss_ramp', type=int, default=20000)
    parser.add_argument('--borders', type=str, choices=['no_edge','black','random','randomrgb','image','white'], default='black')
    parser.add_argument('--y_scale', type=float, default=100.0)
    parser.add_argument('--u_scale', type=float, default=1.0)
    parser.add_argument('--v_scale', type=float, default=1.0)
    parser.add_argument('--adv_y_scale', type=float, default=100.0)
    parser.add_argument('--adv_u_scale', type=float, default=1.0)
    parser.add_argument('--adv_v_scale', type=float, default=1.0)
    parser.add_argument('--no_gan', action='store_true')
    parser.add_argument('--no_lpips', action='store_true')
    parser.add_argument('--no_adv_lpips', action='store_true')
    parser.add_argument('--no_adv_l2', action='store_true')
    parser.add_argument('--rnd_trans', type=float, default=.1)
    parser.add_argument('--rnd_bri', type=float, default=.3)
    parser.add_argument('--rnd_noise', type=float, default=.02)
    parser.add_argument('--rnd_sat', type=float, default=1.0)
    parser.add_argument('--rnd_hue', type=float, default=.1)
    parser.add_argument('--contrast_low', type=float, default=.5)
    parser.add_argument('--contrast_high', type=float, default=1.5)
    parser.add_argument('--jpeg_quality', type=float, default=25)
    parser.add_argument('--no_jpeg', action='store_true')
    parser.add_argument('--rnd_trans_ramp', type=int, default=10000)
    parser.add_argument('--rnd_bri_ramp', type=int, default=1000)
    parser.add_argument('--rnd_sat_ramp', type=int, default=1000)
    parser.add_argument('--rnd_hue_ramp', type=int, default=1000)
    parser.add_argument('--rnd_noise_ramp', type=int, default=1000)
    parser.add_argument('--contrast_ramp', type=int, default=1000)
    parser.add_argument('--jpeg_quality_ramp', type=float, default=1000)
    parser.add_argument('--no_im_loss_steps', help="Train without image loss for first x steps", type=int, default=500)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument("--cuda", type=str, default=0)
    parser.add_argument("--no_wrap", action='store_true')
    parser.add_argument("--num_iter", type=int, default=5)
    parser.add_argument("--attacker_ramp_down", action='store_true')
    parser.add_argument("--attacker_ramp_down_scale_l2", type=float, default=1)
    parser.add_argument("--attacker_ramp_down_scale_lpips", type=float, default=1)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument('--no_attack', action='store_true')
    parser.add_argument("--l2_loss_await", type=int, default=0)
    parser.add_argument("--secret_acc_to_start_img_loss", type=float, default=0.9)
    parser.add_argument('--no_blur_attack', action='store_true')
    parser.add_argument('--blur_attack_kernel_size', type=int, default=15)
    parser.add_argument('--blur_attack_sig', type=float, default=1.)
    # parser.add_argument('--adv_layer_2_kernel_size', type=int, default=3)
    args = parser.parse_args()

    tf.set_random_seed(args.global_seed)

    EXP_NAME = args.exp_name

    files_list = glob.glob(join(TRAIN_PATH,"*"))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    # device = torch.device("cuda")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # def set_seeds(seed=args.global_seed):
    #     os.environ['PYTHONHASHSEED'] = str(seed)
    #     random.seed(seed)
    #     tf.compat.v1.set_random_seed(seed)
    #     np.random.seed(seed)

    # def set_global_determinism(seed=args.global_seed):
    #     set_seeds(seed=seed)

    #     os.environ['TF_DETERMINISTIC_OPS'] = '1'
    #     os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        
    #     tf.config.threading.set_inter_op_parallelism_threads(1)
    #     tf.config.threading.set_intra_op_parallelism_threads(1)

    # # Call the above function with seed value
    # set_global_determinism(seed=args.global_seed)


    height = 128
    width = 128

    secret_pl = tf.placeholder(shape=[None,args.secret_size],dtype=tf.float32,name="input_prep")
    image_pl = tf.placeholder(shape=[None,height,width,3],dtype=tf.float32,name="input_hide")
    M_pl = tf.placeholder(shape=[None,2,8],dtype=tf.float32,name="input_transform")
    global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
    loss_scales_pl = tf.placeholder(shape=[8],dtype=tf.float32,name="input_loss_scales")
    l2_edge_gain_pl = tf.placeholder(shape=[1],dtype=tf.float32,name="input_edge_gain")
    yuv_scales_pl = tf.placeholder(shape=[6],dtype=tf.float32,name="input_yuv_scales")

    log_decode_mod_pl = tf.placeholder(shape=[],dtype=tf.float32,name="input_log_decode_mod")

    encoder = models_AF.StegaStampEncoder(height=height, width=width)
    decoder = models_AF.StegaStampDecoder(secret_size=args.secret_size, height=height, width=width)
    discriminator = models_AF.Discriminator()
    attacker = models_AF.Attacker()

    loss_op, attacked_loss_op, secret_loss_op, D_loss_op, summary_op, image_summary_op, bit_acc = models_AF.build_model(
            encoder=encoder,
            decoder=decoder,
            discriminator=discriminator,
            attacker=attacker,
            secret_input=secret_pl,
            image_input=image_pl,
            l2_edge_gain=l2_edge_gain_pl,
            borders=args.borders,
            secret_size=args.secret_size,
            M=M_pl,
            loss_scales=loss_scales_pl,
            yuv_scales=yuv_scales_pl,
            args=args,
            global_step=global_step_tensor)

    tvars=tf.trainable_variables()  #returns all variables created(the two variable scopes) and makes trainable true


    d_vars=[var for var in tvars if 'discriminator' in var.name]
    g_vars=[var for var in tvars if 'stega_stamp' in var.name]
    adv_vars=[var for var in tvars if 'attacker' in var.name]

    clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]

    train_op = tf.train.AdamOptimizer(args.lr).minimize(loss_op, var_list=g_vars, global_step=global_step_tensor)
    train_adv_op = tf.train.AdamOptimizer(args.adv_lr).minimize(attacked_loss_op, var_list=adv_vars)
    train_secret_op = tf.train.AdamOptimizer(args.lr).minimize(secret_loss_op, var_list=g_vars, global_step=global_step_tensor)
    optimizer = tf.train.RMSPropOptimizer(.00001)
    gvs = optimizer.compute_gradients(D_loss_op, var_list=d_vars)
    capped_gvs = [(tf.clip_by_value(grad, -.25, .25), var) for grad, var in gvs]
    train_dis_op = optimizer.apply_gradients(capped_gvs)

    deploy_hide_image_op, residual_op = models_AF.prepare_deployment_hiding_graph(encoder, secret_pl, image_pl)
    deploy_decoder_op =  models_AF.prepare_deployment_reveal_graph(decoder, image_pl)

    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=100, keep_checkpoint_every_n_hours=4)
    sess.run(tf.global_variables_initializer())
    
    if args.pretrained is not None:
        saver.restore(sess, tf.train.latest_checkpoint(args.pretrained))
        # saver.restore(sess, args.pretrained)

    writer = tf.summary.FileWriter(join(LOGS_Path,EXP_NAME),sess.graph)

    total_steps = len(files_list)//args.batch_size + 1
    global_step = 0
    steps_since_bit_acc_is_enough = 0
    no_im_loss = True

    while global_step < args.num_steps:
        for _ in range(min(total_steps,args.num_steps-global_step)):
            # no_im_loss = global_step < args.no_im_loss_steps
            images, secrets = get_img_batch(files_list=files_list,
                                                     secret_size=args.secret_size,
                                                     batch_size=args.batch_size,
                                                     size=(height,width))
            adv_images, adv_secrets = get_img_batch(files_list=files_list,
                                                     secret_size=args.secret_size,
                                                     batch_size=args.adv_batch_size,
                                                     size=(height,width))
            l2_loss_scale = 0
            lpips_loss_scale = 0
            G_loss_scale = 0

            # if global_step > args.no_im_loss_steps:
            if no_im_loss is False:
                # l2_loss_scale = min(args.l2_loss_scale * (global_step-args.no_im_loss_steps) / (args.l2_loss_ramp-args.no_im_loss_steps), args.l2_loss_scale)
                # lpips_loss_scale = min(args.lpips_loss_scale * (global_step-args.no_im_loss_steps) / (args.lpips_loss_ramp-args.no_im_loss_steps), args.lpips_loss_scale)
                # G_loss_scale = min(args.G_loss_scale * (global_step-args.no_im_loss_steps) / (args.G_loss_ramp-args.no_im_loss_steps), args.G_loss_scale)    
                l2_loss_scale = min(max(0,args.l2_loss_scale * (global_step - (steps_since_bit_acc_is_enough + args.l2_loss_await)) / args.l2_loss_ramp), args.l2_loss_scale)
                lpips_loss_scale = min(max(0,args.lpips_loss_scale * (global_step - (steps_since_bit_acc_is_enough + args.l2_loss_await)) / args.lpips_loss_ramp), args.lpips_loss_scale)
                G_loss_scale = min(max(0,args.G_loss_scale * (global_step - (steps_since_bit_acc_is_enough + args.l2_loss_await)) / args.G_loss_ramp), args.G_loss_scale)
                
            secret_loss_scale = min(args.secret_loss_scale * global_step / args.secret_loss_ramp, args.secret_loss_scale)
            adv_W_secret_loss_scale = min(args.adv_W_secret_loss_scale * global_step / args.adv_W_secret_loss_ramp, args.adv_W_secret_loss_scale)
            if args.attacker_ramp_down is True:
                adv_l2_loss_scale = max(args.adv_l2_loss_scale * (args.attacker_ramp_down_scale_l2 - (args.attacker_ramp_down_scale_l2 - 1) * global_step / args.adv_l2_loss_ramp), args.adv_l2_loss_scale)
                adv_lpips_loss_scale = max(args.adv_lpips_loss_scale * (args.attacker_ramp_down_scale_lpips - (args.attacker_ramp_down_scale_lpips - 1) * global_step / args.adv_lpips_loss_ramp), args.adv_lpips_loss_scale)
            else:
                adv_l2_loss_scale = min(args.adv_l2_loss_scale * global_step / args.adv_l2_loss_ramp, args.adv_l2_loss_scale)
                adv_lpips_loss_scale = min(args.adv_lpips_loss_scale * global_step / args.adv_lpips_loss_ramp, args.adv_lpips_loss_scale)
            
            adv_secret_loss_scale = min(args.adv_secret_loss_scale * global_step / args.adv_secret_loss_ramp, args.adv_secret_loss_scale)
            l2_edge_gain = 0
            if global_step > args.l2_edge_delay:
                l2_edge_gain = min(args.l2_edge_gain * (global_step-args.l2_edge_delay) / args.l2_edge_ramp, args.l2_edge_gain)

            rnd_tran = min(args.rnd_trans * global_step / args.rnd_trans_ramp, args.rnd_trans)
            rnd_tran = np.random.uniform() * rnd_tran
            M = utils.get_rand_transform_matrix(width, np.floor(width * rnd_tran), args.batch_size)

            feed_dict = {secret_pl:secrets,
                         image_pl:images,
                         M_pl:M,
                         l2_edge_gain_pl:[l2_edge_gain],
                         loss_scales_pl:[l2_loss_scale, lpips_loss_scale, secret_loss_scale, G_loss_scale, adv_l2_loss_scale, adv_lpips_loss_scale, adv_secret_loss_scale, adv_W_secret_loss_scale],
                         yuv_scales_pl:[args.y_scale, args.u_scale, args.v_scale, args.adv_y_scale, args.adv_u_scale, args.adv_v_scale],}
            adv_feed_dict = {secret_pl:adv_secrets,
                         image_pl:adv_images,
                         M_pl:M,
                         l2_edge_gain_pl:[l2_edge_gain],
                         loss_scales_pl:[l2_loss_scale, lpips_loss_scale, secret_loss_scale, G_loss_scale, adv_l2_loss_scale, adv_lpips_loss_scale, adv_secret_loss_scale, adv_W_secret_loss_scale],
                         yuv_scales_pl:[args.y_scale, args.u_scale, args.v_scale, args.adv_y_scale, args.adv_u_scale, args.adv_v_scale],}

            # if no_im_loss:
            #     _, _, global_step = sess.run([train_secret_op,loss_op,global_step_tensor],feed_dict)
            # else:
            #     for num_iter in range(args.num_iter):
            #         sess.run([train_adv_op, attacked_loss_op],feed_dict)
            #     _, _, global_step = sess.run([train_op,loss_op,global_step_tensor],feed_dict)
            #     if not args.no_gan:
            #         sess.run([train_dis_op, clip_D],feed_dict)

            for num_iter in range(args.num_iter):
                    sess.run([train_adv_op, attacked_loss_op],adv_feed_dict)
            if no_im_loss:
                _, _, bit_acc_val, global_step = sess.run([train_secret_op,loss_op, bit_acc, global_step_tensor],feed_dict)
                if bit_acc_val > args.secret_acc_to_start_img_loss:
                    steps_since_bit_acc_is_enough = global_step
                    no_im_loss = False
            else:
                _, _, global_step = sess.run([train_op,loss_op,global_step_tensor],feed_dict)
                if not args.no_gan:
                    sess.run([train_dis_op, clip_D],feed_dict)

            if global_step % 100 ==0 :
                summary, global_step = sess.run([summary_op,global_step_tensor], feed_dict)
                writer.add_summary(summary, global_step)
                summary = tf.Summary(value=[tf.Summary.Value(tag='transformer/rnd_tran', simple_value=rnd_tran),
                                            tf.Summary.Value(tag='loss_scales/l2_loss_scale', simple_value=l2_loss_scale),
                                            tf.Summary.Value(tag='loss_scales/adv_l2_loss_scale', simple_value=adv_l2_loss_scale),
                                            tf.Summary.Value(tag='loss_scales/lpips_loss_scale', simple_value=lpips_loss_scale),
                                            tf.Summary.Value(tag='loss_scales/adv_lpips_loss_scale', simple_value=adv_lpips_loss_scale),
                                            tf.Summary.Value(tag='loss_scales/secret_loss_scale', simple_value=secret_loss_scale),
                                            tf.Summary.Value(tag='loss_scales/adv_secret_loss_scale', simple_value=adv_secret_loss_scale),
                                            tf.Summary.Value(tag='loss_scales/adv_W_secret_loss_scale', simple_value=adv_W_secret_loss_scale),
                                            # tf.Summary.Value(tag='loss_scales/y_scale', simple_value=args.y_scale),
                                            # tf.Summary.Value(tag='loss_scales/u_scale', simple_value=args.u_scale),
                                            # tf.Summary.Value(tag='loss_scales/v_scale', simple_value=args.v_scale),
                                            # tf.Summary.Value(tag='loss_scales/adv_y_scale', simple_value=args.adv_y_scale),
                                            # tf.Summary.Value(tag='loss_scales/adv_u_scale', simple_value=args.adv_u_scale),
                                            # tf.Summary.Value(tag='loss_scales/adv_v_scale', simple_value=args.adv_v_scale),
                                            tf.Summary.Value(tag='loss_scales/G_loss_scale', simple_value=G_loss_scale),
                                            tf.Summary.Value(tag='loss_scales/L2_edge_gain', simple_value=l2_edge_gain),
                                            tf.Summary.Value(tag='test/steps_since_bit_acc_is_enough', simple_value=steps_since_bit_acc_is_enough),
                                            tf.Summary.Value(tag='test/global_step', simple_value=global_step),
                                            ])
                writer.add_summary(summary, global_step)

            if global_step % 100 ==0 :
                summary, global_step = sess.run([image_summary_op,global_step_tensor], feed_dict)
                writer.add_summary(summary, global_step)

            if global_step % 10000 ==0:
                save_path = saver.save(sess, join(CHECKPOINTS_PATH,EXP_NAME,EXP_NAME+".chkp"), global_step=global_step)

    constant_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph.as_graph_def(),
            [deploy_hide_image_op.name[:-2], residual_op.name[:-2], deploy_decoder_op.name[:-2]])
    with tf.Session(graph=tf.Graph()) as session:
        # tf.set_random_seed(args.global_seed)
        tf.import_graph_def(constant_graph_def, name='')
        tf.saved_model.simple_save(session,
                                   SAVED_MODELS + '/' + EXP_NAME,
                                   inputs={'secret':secret_pl, 'image':image_pl},
                                   outputs={'stegastamp':deploy_hide_image_op, 'residual':residual_op, 'decoded':deploy_decoder_op})

    writer.close()

if __name__ == "__main__":
    main()
