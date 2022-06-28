import io

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from starlette.responses import StreamingResponse

from predict import InferenceSeg
# import imantics


app = FastAPI(
    title="Segmentation Model",
    description="Api for segmentation model training and inference",
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
detector = InferenceSeg('mobilenet_v2', 'imagenet')


@app.get("/")
def root():
    return {"message": "Welcome to the Segmentation API get documentation at /docs"}


@app.post("/predict")
async def predict(image: UploadFile = File(...), return_image: bool = False, return_coco: bool = False,
                  ):
    # try:
        if return_image and return_coco:
            raise ValueError('return_image and return_coco cannot be True at the same time')

        contents = await image.read()
        nparr = np.fromstring(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        response = detector.predict_data(img, return_image, return_coco)
        if return_image:
            return StreamingResponse(io.BytesIO(response), media_type='image/jpeg')

        elif return_coco:
            print('yesss')
            # m, classes = response
            return {"results": response}
        return {"message": "Success"}

    # except Exception as e:
    #     return {"message": str(e)}

@app.post("/set_classes")
async def set_classes(n_classes: int):
    detector.set_classes(n_classes)
    return {"message": "Success"}


@app.post("/set_size")
async def set_size(weight: int, height: int):
    detector.set_size(weight, height)
    return {"message": "Success"}


@app.post("/set_model")
async def set_model(model_path: str):
    detector.set_model(model_path)
    return {"message": "Success"}