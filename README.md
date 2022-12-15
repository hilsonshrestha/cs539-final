# ReadMe

# Image caption generator with transformer

## Datasets

This project uses coco dataset.

Download and unzip dataset to `coco_dataset` directory:

```sh
mkdir coco_dataset

wget -O coco_dataset/train2014.zip http://images.cocodataset.org/zips/train2014.zip
unzip 'coco_dataset/train2014.zip'
rm -r 'coco_dataset/train2014.zip'

wget -O coco_dataset/val2014.zip http://images.cocodataset.org/zips/val2014.zip
unzip 'coco_dataset/val2014.zip'
rm -r 'coco_dataset/val2014.zip'

wget -O coco_dataset/caption_datasets.zip http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
unzip 'coco_dataset/caption_datasets.zip'
rm -r 'coco_dataset/caption_datasets.zip'
```

## Setup

### Create a virtual environment and install dependencies in it.
```
virtualenv venv --python=/usr/bin/python3
source venv/bin/activate
pip install -r requirements.txt
```

## Run

To run the transformer:
```
cd cnn_transformer
python transformer.py
```

# m2_transformer
## Environment setup

The concerned files are in the m2_transformer folder. Create the `m2release` conda environment using the `environment.yml` file.

Download spacy data by executing the following command:
```
python -m spacy download en
```
## Data preparation

Please download the COCO annotations file [annotations.zip](https://drive.google.com/file/d/1i8mqKFKhqvBr8kEp3DbIh9-9UNAfKGmE/view?usp=sharing) and extract it.

Detection features are computed from feature representations of Faster RCNN object detectiions. To obtain precomputed feature representations of COCO dataset, please download the COCO features file [coco_detections.hdf5](https://drive.google.com/open?id=1MV6dSnqViQfyvgyHrmAT_lLpFbkzp3mx) (~53.5 GB), in which detections of each image are stored under the `<image_id>_features` key. `<image_id>` is the id of each COCO image, without leading zeros.

## Run code
Run `python train.py` or `python test.py` with respective arguments to train or test model. Pretrained model is available at https://drive.google.com/file/d/1tPFbzfmPp56mrRE43BweJD7CzP5w5pxl/view?usp=share_link

Use `python custom.py` to test model on specific images of COCO dataset

