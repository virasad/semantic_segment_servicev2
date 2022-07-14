# semantic_segment_servicev2

Train Classification model as a service. **Easy as ABC!**

The goal of this project is to train and test a classification model without any code or knowledge of the deeplearning.

# Usage

## Docker Compose

### Install dependencies

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- Docker gpu

```bash
# It should be install for gpu support
bash docker-gpu-installation.sh
```

```bash
docker-compose up -d
```

---

## Shell

### Install dependencies

- [Pytorch & TorchVision](https://pytorch.org/get-started/locally/)

```bash
cd train
pip install -r requirements.txt

cd inference
pip install -r requirements.txt
```

---

# Train

## Prepare data

Put your dataset in the following format:
- If we have coco dataset

```bash
- dataset
  - images
    - image.jpg
    - image.jpg
    - image.jpg
      ...
      
  {COCOLABELS}.json
  
      
```

## Train from bash

```bash
cd train
python trainer.py --images_dir {YOUR_DATASET_PATH} \
                  --masks_dir {YOUR_DATASET_PATH} \ # If you have mask dataset masks_dir and if not, coco dataset set {COCOLABELS}.json
                  --model_name {MODEL_NAME} \
                  --w_size {WIDTH} \
                  --h_size {HEIGHT} \
                  --n_classes {NUMBER_OF_CLASSES} \ 
                  --response_url {RESPONSE_URL} \ # If you want to send response to send final result to your server (optional)
                  --log_url {LOG_URL} \ # If you want to send log to your server (optional)
                  --epochs {EPOCHS} \
                  --batch_size {BATCH_SIZE} \
                  --num_dataloader_workers {NUM_DATALOADER_WORKERS} \
                  --image_width {NETWORK_INPUT_WIDTH} \
                  --image_height {NETWORK_INPUT_HEIGHT} \
                  --validation_split {VALIDATION_SPLIT} \
                  --pretrained_path {PRETRAINED_PATH} \
                  --backbone {BACKBONE} 
```

## Train with API

If you run the docker compose file you should put your dataset in volumes/dataset folder and your weights in
volumes/weights folder.

### Parameters

```json
{
  "dataset_p": dataset_p,
  "save_name": save_name,
  "batch_size": batch_size,
  "num_dataloader_workers": num_dataloader_workers,
  "epochs": epochs,
  "validation_split": validation_split,
  "pretrained_path": pretrained_path,
  "backbone": backbone,
  "image_width": image_width,
  "image_height": image_height,
  "response_url": response_url,
  "extra_kwargs": extra_kwargs
}
```

- **response_url** is the url to send the response. It can be Null or not send

- **extra_kwargs** is a dictionary that will send back to your response API after the train is finished.

### Example

- For train example refer to the [example](train/test.py)
- For inference example refer to the [example](inference/test.py)
