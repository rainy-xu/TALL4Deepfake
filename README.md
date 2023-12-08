# Official implementation for TALL
[ICCV2023-TALL: Thumbnail Layout for Deepfake Video Detection](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_TALL_Thumbnail_Layout_for_Deepfake_Video_Detection_ICCV_2023_paper.pdf)

Our implementation is based on [Swin-Transformer](https://github.com/microsoft/Swin-Transformer).

# Requirements
 - einops
 - fvcore
 - timm==0.4.12
 - torch==1.13.1
 - torchaudio==0.13.1
 - torchvision==0.14.1

# Data Preparation
Please refer to https://github.com/IBM/action-recognition-pytorch for how to prepare deepfake datasets such as FF++, Celeb-DF, and DFDC.

# Training:

[IMPORTANT] Edit main.py and change the default arg-parser values according to your convenience (especially the config paths)

CUDA_VISIBLE_DEVICES=0 python main.py --dataset ffpp \
 --input-size 112 --num_clips 8 --output_dir [your_output_dir] --opt adamw --lr 1.5e-5 --warmup-lr 1.5e-8 --min-lr 1.5e-7 \
 --epochs 60 --sched cosine --duration 4 --batch-size 4 --thumbnail_rows 2 --disable_scaleup --cutout True \
 --pretrained --warmup-epochs 10 --no-amp --model TALL_SWIN \
 --hpe_to_token 2>&1 | tee ./output/train_ffpp_`date +'%m_%d-%H_%M'`.log

# Evaluation:

CUDA_VISIBLE_DEVICES=0 python test.py  --dataset ffpp \
 --input_size 112 --opt adamw --lr 1e-4 --epochs 30 --sched cosine --duration 4 --batch-size 4 --thumbnail_rows 2 --disable_scaleup \
 --pretrained --warmup-epochs 5 --no-amp --model TALL_SWIN  \
 --hpe_to_token --initial_checkpoint [model_checkpoint] --eval --num_crops 1 --num_clips 8 \
 2>&1 | tee ./output/test_ffpp_`date +'%m_%d-%H_%M'`.log

# Citation

@inproceedings{xu2023tall,
  title={TALL: Thumbnail Layout for Deepfake Video Detection},
  author={Xu, Yuting and Liang, Jian and Jia, Gengyun and Yang, Ziming and Zhang, Yanhao and He, Ran},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={22658--22668},
  year={2023}
}

# Contact

- [xuyuting@iie.ia.ac](mailto:xuyuting@iie.ia.ac)
- [liangjian92@gmail.com](mailto:liangjian92@gmail.com)
