import glob
import os
from PIL import Image,ImageOps
import numpy as np
import random
import tensorflow as tf
import utils
import models_mimic_blur
from os.path import join


TRAIN_PATH = './datasets/img_align_celeba_128x128/'
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
    parser.add_argument('--num_steps', type=int, default=80000)
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

    # secret_pl = tf.placeholder(shape=[None,args.secret_size],dtype=tf.float32,name="input_prep")
    image_pl = tf.placeholder(shape=[None,height,width,3],dtype=tf.float32,name="input_hide")
    # M_pl = tf.placeholder(shape=[None,2,8],dtype=tf.float32,name="input_transform")
    global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
    loss_scales_pl = tf.placeholder(shape=[8],dtype=tf.float32,name="input_loss_scales")
    # l2_edge_gain_pl = tf.placeholder(shape=[1],dtype=tf.float32,name="input_edge_gain")
    # yuv_scales_pl = tf.placeholder(shape=[6],dtype=tf.float32,name="input_yuv_scales")

    # log_decode_mod_pl = tf.placeholder(shape=[],dtype=tf.float32,name="input_log_decode_mod")

    # encoder = models_mimic_blur.StegaStampEncoder(height=height, width=width)
    # decoder = models_mimic_blur.StegaStampDecoder(secret_size=args.secret_size, height=height, width=width)
    # discriminator = models_mimic_blur.Discriminator()
    attacker = models_mimic_blur.Attacker()

    loss_op, summary_op, image_summary_op = models_mimic_blur.build_model(
            attacker=attacker,
            image_input=image_pl,
            args=args,
            global_step=global_step_tensor)

    tvars=tf.trainable_variables()  #returns all variables created(the two variable scopes) and makes trainable true


    # d_vars=[var for var in tvars if 'discriminator' in var.name]
    # g_vars=[var for var in tvars if 'stega_stamp' in var.name]
    adv_vars=[var for var in tvars if 'attacker' in var.name]

    # clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]

    # train_op = tf.train.AdamOptimizer(args.lr).minimize(loss_op, var_list=g_vars, global_step=global_step_tensor)
    train_op = tf.train.AdamOptimizer(args.lr).minimize(loss_op, var_list=adv_vars, global_step=global_step_tensor)
    # train_secret_op = tf.train.AdamOptimizer(args.lr).minimize(secret_loss_op, var_list=g_vars, global_step=global_step_tensor)
    # optimizer = tf.train.RMSPropOptimizer(.00001)
    # gvs = optimizer.compute_gradients(D_loss_op, var_list=d_vars)
    # capped_gvs = [(tf.clip_by_value(grad, -.25, .25), var) for grad, var in gvs]
    # train_dis_op = optimizer.apply_gradients(capped_gvs)

    deploy_attacked_image_op, deploy_transformed_image_op, deploy_residual_op = models_mimic_blur.prepare_deployment_graph(attacker, image_pl, args)

    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=100, keep_checkpoint_every_n_hours=4)
    sess.run(tf.global_variables_initializer())
    
    if args.pretrained is not None:
        # saver.restore(sess, tf.train.latest_checkpoint(args.pretrained))
        saver.restore(sess, args.pretrained)

    writer = tf.summary.FileWriter(join(LOGS_Path,EXP_NAME),sess.graph)

    total_steps = len(files_list)//args.batch_size + 1
    global_step = 0

    while global_step < args.num_steps:
        for _ in range(min(total_steps,args.num_steps-global_step)):
            # no_im_loss = global_step < args.no_im_loss_steps
            images, secrets = get_img_batch(files_list=files_list,
                                                     secret_size=args.secret_size,
                                                     batch_size=args.batch_size,
                                                     size=(height,width))

            feed_dict = {image_pl:images,}
            _, _, global_step = sess.run([train_op,loss_op,global_step_tensor],feed_dict)

            if global_step % 100 ==0 :
                summary, global_step = sess.run([summary_op,global_step_tensor], feed_dict)
                writer.add_summary(summary, global_step)

            if global_step % 100 ==0 :
                summary, global_step = sess.run([image_summary_op,global_step_tensor], feed_dict)
                writer.add_summary(summary, global_step)

            if global_step % 10000 ==0:
                save_path = saver.save(sess, join(CHECKPOINTS_PATH,EXP_NAME,EXP_NAME+".chkp"), global_step=global_step)

    constant_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph.as_graph_def(),
            [deploy_attacked_image_op.name[:-2], deploy_transformed_image_op.name[:-2], deploy_residual_op.name[:-2]])
    with tf.Session(graph=tf.Graph()) as session:
        # tf.set_random_seed(args.global_seed)
        tf.import_graph_def(constant_graph_def, name='')
        tf.saved_model.simple_save(session,
                                   SAVED_MODELS + '/' + EXP_NAME,
                                   inputs={'image':image_pl},
                                   outputs={'attacked':deploy_attacked_image_op, 'transformed':deploy_transformed_image_op, 'residual':deploy_residual_op})

    writer.close()

if __name__ == "__main__":
    main()