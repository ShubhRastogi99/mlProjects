from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

Model1 = load_model('model.hdf5')
Model2 = load_model('model2.hdf5')
Model3 = load_model('model3.hdf5')
Model4 = load_model('model4.hdf5')
Model5 = load_model('model5.hdf5')
models = [Model1, Model2, Model3, Model4, Model5]
weights = [0.1, 0.23, 0.17, 0.25, 0.25]

Class_names = ['No Tumor', 'Tumor']
Image_shape = [224, 224]


@app.get("/ping")
async def ping():
    return "Hello"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image = image/255
    image = cv2.resize(image, Image_shape, interpolation=cv2.INTER_CUBIC)

    if len(image.shape) == 2:
        image = tf.expand_dims(image, -1)
        x1 = image.shape[0]
        x2 = image.shape[1]
        image = tf.reshape(tf.broadcast_to(image, (x1, x2, 3)),  (x1, x2, 3))

    img_batch = np.expand_dims(image, axis=0)

    preds = [model.predict(img_batch) for model in models]
    preds = np.array(preds)

    weighted_pred = np.tensordot(preds, weights, axes=((0), (0)))
    weighted_ensemble_preds = np.argmax(weighted_pred[0])

    predicted_class = Class_names[weighted_ensemble_preds]
    confidence = np.max(weighted_pred[0])
    confidence = confidence.tolist()

    return {
        'class': predicted_class,
        'confidence': confidence
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=5000)
