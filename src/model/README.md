# Model

## Install SA-FARI dataset

[URL](https://console.cloud.google.com/storage/browser/cxl-public-camera-trap/sa_fari/)

- Download Images

```bash
cd src/model && mkdir data
# SA-FARI Test / Train Images
wget https://storage.googleapis.com/cxl-public-camera-trap/sa_fari/sa_fari_test_tars/JPEGImages_6fps/sa_fari_test_00.tar.gz
tar vxf sa_fari_test_00.tar.gz
wget https://storage.googleapis.com/cxl-public-camera-trap/sa_fari/sa_fari_train_tars/JPEGImages_6fps/sa_fari_train_00.tar.gz
tar vxf sa_fari_train_00.tar.gz
```

- Download annotations

```bash
cd src/model
# SA-FARI Test / Train Annoations
wget https://huggingface.co/datasets/facebook/SA-FARI/blob/main/annotation/sa_fari_test.json
wget https://huggingface.co/datasets/facebook/SA-FARI/blob/main/annotation/sa_fari_test_ext.json
wget https://huggingface.co/datasets/facebook/SA-FARI/blob/main/annotation/sa_fari_train.json
wget https://huggingface.co/datasets/facebook/SA-FARI/blob/main/annotation/sa_fari_train_ext.json
```

## Generate embeddings

This will use a DINOV2 model and generates emebeddings to be used for SSL training

```bash
uv sync
source ./.venv/bin/activate
python3 extract_features.py
```

## Training SSL

This will train the SSL model to learn the DINO emebdedding

```bash
python3 train.py
```

## Finetuning SSL

This will finetune the backbone on a classification downstream task.

```bash
python3 finetune.py
```
