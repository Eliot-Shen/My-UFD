CUDA_VISIBLE_DEVICES=4                              \
python train.py --name=clip_vitl14_336                      \
--real_list_path=/home/data/szk/Fakeclub/ufd/0_real         \
--fake_list_path=/home/data/szk/Fakeclub/ufd/1_fake          \
--batch_size=16                                        \
--suffix=time                                            \
--data_mode=ours  \
--arch=CLIP:ViT-L/14@336px  \
--fix_backbone  \
--lr=0.0001
