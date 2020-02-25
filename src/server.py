# Original imports 
import logging
from io import BytesIO
import os
import cv2
import msgpack
import pydicom
import numpy as np
#import tensorflow as tf
from flask import Flask, Response, abort, request
from PIL import Image
#from tf_explain.core import GradCAM, SmoothGrad
from waitress import serve
#

import torch
import yaml
from mmcv.parallel import collate 

from factory import set_reproducibility

import factory.evaluate as evaluate
import factory.builder as builder
import factory.models as models


app = Flask(__name__)
model = None

with open('configs/experiment001.yaml') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
#config = {"input_shape": (224, 224, 3)}


def load_model():
    global model
    model = builder.build_model(cfg, 0)
    model.load_state_dict(torch.load(cfg['predict']['checkpoint'], map_location=lambda storage, loc: storage))
    model = model.eval()


def prepare_input(image):
    # Need to save temporary image file
    rand_img = 'img-{}.png'.format(np.random.randint(1e10))
    # Assuming image is (H, W)
    if image.ndim == 2:
        X = np.expand_dims(image, axis=-1)
        X = np.repeat(X, 3, axis=-1)
    else:
        X = image
    cv2.imwrite(rand_img, X)
    ann = [{
        'filename': rand_img,
        'height': X.shape[0],
        'width':  X.shape[1]
    }]
    cfg['dataset']['data_dir'] = '.'
    loader = builder.build_dataloader(cfg, ann, mode='predict')
    data = loader.dataset[0]
    img = data.pop('img')
    img_meta = {0 : data}
    os.system('rm {}'.format(rand_img))
    return img, img_meta


def predict(x):
    img, img_meta = prepare_input(x)
    with torch.no_grad():
        output = model([img.unsqueeze(0)], img_meta=[img_meta], return_loss=False, rescale=True)
    output = output[0]
    threshold = 0.3 # adjust as necessary
    output = output[output[:,-1] >= threshold]
    x1 = output[:,0]
    x2 = output[:,1]
    y1 = output[:,2]
    y2 = output[:,3]
    p  = output[:,4]
    result = ...
    return result


@app.route("/inference", methods=["POST"])
def inference():
    """
    Route for model inference.

    The POST body is msgpack-serialized binary data with the follow schema:

    {
        "instances": [
            {
                "file": "bytes"
                "tags": {
                    "StudyInstanceUID": "str",
                    "SeriesInstanceUID": "str",
                    "SOPInstanceUID": "str",
                    ...
                }
            },
            ...
        ],
        "args": {
            "arg1": "str",
            "arg2": "str",
            ...
        }
    }

    The `file bytes is the raw binary data representing a DICOM file, and can be loaded using
    `pydicom.dcmread()`.

    The response body should be the msgpack-serialized binary data of the results:

    [
        {
            "study_uid": "str",
            "series_uid": "str",
            "instance_uid": "str",
            "frame_number": "int",
            "class_index": "int",
            "data": {},
            "probability": "float",
            "explanations": [
                {
                    "name": "str",
                    "description": "str",
                    "content": "bytes",
                    "content_type": "str",
                },
                ...
            ],
        },
        ...
    ]
    """
    if not request.content_type == "application/msgpack":
        abort(400)

    data = msgpack.unpackb(request.get_data(), raw=False)
    input_instances = data["instances"]
    input_args = data["args"]

    results = []

    for instance in input_instances:
        try:
            tags = instance["tags"]
            ds = pydicom.dcmread(BytesIO(instance["file"]))
            image = ds.pixel_array
            result = predict(prepare_input(image))
            result["study_uid"] = tags["StudyInstanceUID"]
            result["series_uid"] = tags["SeriesInstanceUID"]
            result["instance_uid"] = tags["SOPInstanceUID"]
            result["frame_number"] = None
            results.append(result)
        except Exception as e:
            logging.exception(e)
            abort(500)

    resp = Response(msgpack.packb(results, use_bin_type=True))
    resp.headers["Content-Type"] = "application/msgpack"
    return resp


@app.route("/healthz", methods=["GET"])
def healthz():
    return "", 200


if __name__ == "__main__":
    load_model()
    serve(app, listen="*:6324")