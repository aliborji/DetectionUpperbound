# to test
# python train_net.py --config ./configs/markable.yaml -a resnet101 --batch-size 250 --pretrained --num-workers 10 --multiprocessing-distributed --rank 0 --world-size 1 --evaluate --resume ./output/markable/1/model_last.pth #./output/voc/0.2/checkpoint_14.pth

# python train_net.py --config ./configs/coco.yaml     -a resnet101 --pretrained --epochs 14 --batch-size 50 --num-workers 10 --lr 0.005 --multiprocessing-distributed --rank 0 --world-size 1  #--evaluate #--resume ./output/voc/checkpoint_9.pth



# for debugging only
# CUDA_VISIBLE_DEVICES=0 python train_net.py --config ./configs/coco.yaml -a resnet101 --batch-size 250 --pretrained --evaluate --resume ./output/coco/1/checkpoint_3.pth #./output/voc/0.2/checkpoint_14.pth

# python train_net.py --config ./configs/coco.yaml -a resnet101 --batch-size 50 --pretrained --num-workers 10 --evaluate --resume ./output/coco/1/checkpoint_3.pth #./output/voc/0.2/checkpoint_14.pth


# python train_net.py --config ./configs/markable.yaml -a resnet152 --batch-size 50 --pretrained --num-workers 20 --evaluate --resume ./output/markable/1_new/checkpoint_9.pth


# python train_net.py --config ./configs/coco.yaml -a resnet152 --batch-size 50 --pretrained --num-workers 10 --evaluate --resume ./output_context/coco/1.8/model_last.pth


# VOC  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
python train_net.py --config ./configs/voc.yaml -a resnet101 --batch-size 50 --pretrained --num-workers 10 --evaluate --resume ./output/voc/1/model_last.pth
# python train_net.py --config ./configs/voc.yaml -a resnet101 --batch-size 50 --pretrained --num-workers 10 --evaluate --resume ./output_context/voc/5/checkpoint_13.pth
# python train_net.py --config ./configs/voc.yaml -a resnet101 --batch-size 50 --pretrained --num-workers 10 --evaluate --resume ./output_extended/voc/1/model_last.pth
# 




# COCO  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# python train_net.py --config ./configs/coco.yaml -a resnet101 --batch-size 50 --pretrained --num-workers 10 --evaluate --resume ./output_context/coco/2/checkpoint_9.pth
# python train_net.py --config ./configs/coco.yaml -a resnet101 --batch-size 50 --pretrained --num-workers 10 --evaluate --resume ./output_context/coco/10/checkpoint_4.pth
# python train_net.py --config ./configs/coco.yaml -a resnet101 --batch-size 50 --pretrained --num-workers 10 --evaluate --resume ./output_extended/coco/1/model_last.pth



# MARKABLE  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# python train_net.py --config ./configs/markable.yaml -a resnet101 --batch-size 50 --pretrained --num-workers 10 --evaluate --resume ./output/markable/1.2/checkpoint_13.pth

# python train_net.py --config ./configs/markable.yaml -a resnet101 --batch-size 50 --pretrained --num-workers 10 --evaluate --resume ./output_context/markable/1.2/checkpoint_13.pth

# python train_net.py --config ./configs/openImages.yaml -a resnet101 --batch-size 50 --pretrained --num-workers 10 --evaluate --resume ./output/openImages/1/checkpoint_4.pth


#python train_net.py --config ./configs/voc.yaml -a resnet101 --batch-size 50 --pretrained --num-workers 10 --evaluate --resume ./output/voc/1/model_last.pth

# python train_net.py --config ./configs/markable.yaml -a resnet101 --batch-size 50 --pretrained --num-workers 10 --evaluate --resume ./output_extended/markable/1/model_last.pth