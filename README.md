# DCCN
The code for "Dynamic Context-guided Capsule Network for Multimodal Machine Translation" in Pytorch. 
This project is from https://github.com/DeepLearnXMU/MM-DCCN/.

## Installation

Tested with Python 3.6, CUDA 10.2, Pytorch 1.7, Open Suse Leap 15.2.

Install from the requirements file.
```shell
pip install -r requirements.txt
# Or, alternatively
pip install torch torchvision torchaudio torchtext sacrebleu opencv-python requests pycocotools ray fvcore
```

Install object detection code.
```shell
cd bottom-up-attention.pytorch/detectron2
pip install -e .
cd ../apex
pip install -e .
cd ..
pip install -e .
cd ..
```

Install MM-DCCN code.
```shell
pip install -e .
```

Download pre-trained global image feature (ResNet-50) model from [here](https://download.pytorch.org/models/resnet50-0676ba61.pth), download pre-trained objection detection feature (Faster R-CNN with ResNet-101 backbone) model [here](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EaXvCC3WjtlLvvEfLr3oa8UBLA21tcLh4L8YLbYXl6jgjg?download=1), and copy into `work/` directory.


## Quickstart

### Download Multi30k dataset and tokenize

Download the WMT 2018 multimodal translation task version of Multi30k:
```shell
git clone --recursive https://github.com/vvjn/multi30k-wmt18
```
The images in the Multi30k dataset are from the Flickr30k dataset, the images of which can be downloaded from [here](http://shannon.cs.illinois.edu/DenotationGraph/data/index.html), and the `test_2017_flickr` and `test_2018_flickr` images from [here](https://drive.google.com/drive/folders/1kfgmYFL5kup51ET7WQNxYmKCvwz_Hjkt).

Tokenize
```shell
cd multi30k-wmt18/
sh prepare-wmt18-text.sh
```

### Preprocess the image data

Extract global image features.
```shell
cd work/
for x in train val test_2016_flickr; do
    python ../MM-DCCN/extract_global_image_feats.py -i ../data/flickr30k-images -f ../multi30k-wmt18/task1-data/image_splits/$x.txt -m resnet50-0676ba61.pth -o ../multi30k-wmt18/task1-data/features_resnet50/$x
done
```

Extract bottom-up object detection features.
```shell
x=flickr30k-images
python ../MM-DCCN/bottom-up-attention.pytorch/extract_scores_and_features.py --mode caffe --num-cpus 4 --extract-mode roi_feats --min-max-boxes 10,10 --gpus 0 --config-file ../MM-DCCN/bottom-up-attention.pytorch/configs/bua-caffe/extract-bua-caffe-r101.yaml --image-dir ../data/$x --out-dir ../multi30k-wmt18/task1-data/features_butd_m10m10/$x
```

Convert from BU format to DCCN format.
```shell
for x in val train test_2016_flickr; do
    python ../MM-DCCN/convert_butd_feats_to_dccn.py -i ../multi30k-wmt18/task1-data/features_butd_m10m10/flickr30k-images -f ../multi30k-wmt18/task1-data/image_splits/$x.txt  -v onmt-runs/mmod_run3/data/processed.vocab.pt -o ../multi30k-wmt18/task1-data/features_dccn/$x
done
```

### Preprocess/train/translate/evaluate the text data

```shell
cd work/
sh ../MM-DCCN/bpe_pipeline_mmod.sh run1 [preprocess|train|translate|evaluate] ../MM-DCCN ../multi30k-wmt18/task1-data/en-de ../multi30k-wmt18/task1-data/features_resnet50 ../multi30k-wmt18/task1-data/features_dccn
```
