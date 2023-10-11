#! /bin/scripts


python main.py --opt options/MeD_test/test.json \
                --task denoising --gpuid 0 --test  \
                --init_iter 0 \
                --pretrain_dir \
                ../results/denoising





