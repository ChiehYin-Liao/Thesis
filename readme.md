# README
`${EXP_NAME}` is the name of the fingerprinting model. Note that you have to train the model with `train_with_attack_AF.py`.

## `train_with_attack_AF.py`
This file train the proposed fingerprinting model
#### Usage
```
python train_with_attack_AF.py ${EXP_NAME} \
    --secret_acc_to_start_img_loss 0.9 \
    --batch_size 64 \
    --adv_batch_size 64 \
    --blur_attack_kernel_size 15 \
    --no_gan \
    --no_lpips \
    --no_adv_lpips \
    --global_seed 0 \
    --secret_size 100 \
    --num_steps 95000 \
    --l2_loss_await 0 \
    --adv_l2_loss_scale 10000 \
    --adv_secret_loss_scale 1 \
    --num_iter 3 \
    --l2_loss_scale 10 \
    --lpips_loss_scale 0.5 \
    --G_loss_scale 0.5 \
    --secret_loss_scale 0.5 \
    --adv_W_secret_loss_scale 0.5 \
    --adv_l2_loss_ramp 4500 \
    --adv_lpips_loss_ramp 3000  \
    --adv_secret_loss_ramp 3000 \
    --l2_loss_ramp 3000 \
    --lpips_loss_ramp 3000  \
    --G_loss_ramp 3000 \
    --secret_loss_ramp 1 \
    --adv_W_secret_loss_ramp 1 \
    --adv_lr 0.00002 \
    --lr 0.00002 \
    --no_wrap \
    --attacker_ramp_down_scale_l2 4 \
    --attacker_ramp_down_scale_lpips 1 \
    --attacker_ramp_down
```

## `encode_image.py`
This file can be used to encode fingerprints to images
#### Usage
```
python encode_image.py \
    --model ${EXP_NAME}
```

## `decode_image.py`
This file can be used to decode fingerprints from fingerprinted images
#### Usage
```
python decode_image.py \
    --model ${EXP_NAME}
```

## `train_mimic_blur.py`
This file can be use to evaluate the imitative ability of the CNN-based attacker
#### Usage
```
python train_mimic_blur.py ${EXP_NAME} \
    --blur_attack_kernel_size 3 \
    --blur_attack_sig 1.0 \
    --batch_size 64 \
    --global_seed 0 \
    --lr 0.0001
```
