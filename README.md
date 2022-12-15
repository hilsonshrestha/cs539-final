# ReadMe

Image caption generator with transformer

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
python transformer.py
```
