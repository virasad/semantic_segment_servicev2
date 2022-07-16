import os
from enum import Enum

from fastapi import FastAPI, BackgroundTasks

import trainer as tr
from utils.callbacks import RedisCallback


class DataType(str, Enum):
    voc = "voc"
    coco = "coco"
    mask_raw = "mask_raw"


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(
    title="Segmentation Train API",
    description="Api for segmentation model training",
    version="0.1.0",
    contact={
        "name": "Virasad",
        "url": "https://virasad.ir",
        "email": "info@virasad.ir",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)


@app.get("/")
def root():
    return {"message": "Welcome to the Segmentation API get documentation at /docs"}


@app.post("/train")
async def train(
        images_dir: str,
        masks_dir: str,
        n_classes: int,
        batch_size: int,
        epochs: int,
        model_name: str,
        w_size: int,
        h_size: int,
        background_tasks: BackgroundTasks,
        response_url: str = None,
        log_url: str = None,
        task_id: str = None,
        data_type: DataType = DataType.coco,
        redis_cb: bool = True,
        file_cb: bool = False,
):
    # background_tasks.add_task(
    tr.trainer(images_dir=images_dir,
               masks_dir=masks_dir,
               model_name=model_name,
               n_classes=n_classes,
               w_size=w_size,
               h_size=h_size,
               batch_size=batch_size,
               epochs=epochs,
               response_url=response_url,
               log_url=log_url,
               task_id=task_id,
               data_type=data_type,
               file_cb=file_cb,
               redis_cb=redis_cb,
               )

    return {"message": "Success"}


@app.get("/get-status")
async def get_status(task_id: int):
    # with open(os.path.join(ROOT_DIR, 'runs', str(task_id), 'logs', 'result.json')) as f:
    #     return json.load(f)
    response = RedisCallback(task_id, 'redis').get()
    return response
