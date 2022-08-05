python3 main_train_eval.py  \
--K 6 --mode 'Micro' --dataset 'CAS' \
--data_root './datasets/face/emotion/micro_macro_datasets/SPOT'  \
--backbone_type 'MobileNet' --head_type 'MagFace' --bool_head True \
--lr 0.01 --epoches 9 --batch_size 256
