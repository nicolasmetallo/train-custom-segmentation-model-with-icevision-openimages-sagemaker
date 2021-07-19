# Load Dependencies
# ----------------------------------------------------------------------------------

import base64
import io
import logging
import os
import glob
import json
import numpy as np
import torch
import time
import cv2

from PIL import Image, ImageChops
from torch.autograd import Variable
from torchvision import transforms

logger = logging.getLogger(__name__)

# Helper Funcs
# ----------------------------------------------------------------------------------

def _get_transforms():
    """ Resize image, normalise it versus Imagenet, convert to tensor """
    return transforms.Compose(
        [
            transforms.Resize(384), 
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406], 
                std=[0.229,0.224,0.225])
        ])

def _get_device():
    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(
        map_location + ":" + str(properties.get("gpu_id"))
        if torch.cuda.is_available() else map_location)
    logger.info(f"Device on initialisation is: {device}")
    return map_location, device

def _trim_whitespaces(image):
    """ Trim whitespace from PIL image """
    bg = Image.new(image.mode, image.size, image.getpixel((0,0)))
    diff = ImageChops.difference(image, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return image.crop(bbox)

def _transform_pred_img(image, mask_array):
    """ Uses a segmentation mask to remove non-detected areas from an image.

    Args:
        image: input image as numpy array
        mask_array: predicted segmentation mask as numpy array

    Returns: A PIL image.
    """
    thresh = cv2.threshold(mask_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    result = cv2.bitwise_and(image, image, mask=thresh)
    result[thresh==0] = [255,255,255] # Turn background white
    output_img = _trim_whitespaces(Image.fromarray(result))
    return output_img


# Image Segmentation Model Definition
# ----------------------------------------------------------------------------------

from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from typing import *


class ImageSegmenter(MaskRCNN):
    def __init__(self, num_classes=2, **kwargs):
        backbone = resnet_fpn_backbone('resnet50', True)
        super().__init__(backbone, num_classes, **kwargs)

def _noop_normalize(image: torch.Tensor) -> torch.Tensor:
    return image

def _noop_resize(
    image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]]
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    return image, target

def remove_internal_model_transforms(model: GeneralizedRCNN):
    model.transform.normalize = _noop_normalize
    model.transform.resize = _noop_resize


# Main Functions
# ----------------------------------------------------------------------------------

def model_fn(model_dir):
    """
    Loads the pickle file from the given model path.
    Args:
        model_dir (str): Points to the location of the model artefacts.
    Raises:
        RuntimeError: It raises this error when the model artefacts file is missing.

    Returns:
        serialized model file: Returns the pickled pytorch model file
    """
    map_location, device = _get_device()
    model_path = os.path.join(model_dir, 'model.pth')
    if not os.path.isfile(model_path):
        raise RuntimeError("Missing the model artefacts file")
        
    model = ImageSegmenter()
    state_dict = torch.load(model_path, map_location=map_location)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    remove_internal_model_transforms(model) # remove internal transforms (e.g. resize, normalise)
    
    return model
    
def preprocess(image):
    """
    Read PIL image and return torch.tensor.
    """
    image_tfms = _get_transforms()
    map_location, device = _get_device()

    return image_tfms(image).to(device), np.asarray(image)

def inference(self, img):
    """
    Predict segmentation mask of an image using trained DL model
    """
    logger.info(f"Device on inference is: {self.device}")
    self.model.eval()

    with torch.no_grad():
        outputs = self.model([img])
    logger.info(outputs)

    return outputs

def postprocess(self, inference_output, image_as_numpy_array):
    """
    Use segmentation mask to remove the background from your Region of Interest (ROI)
    and trim whitespaces from image.
    """
    image_size = image_as_numpy_array.shape[:2][::-1]
    inference_output = inference_output[1:][0][0] if self.is_model_scripted else inference_output[0]
    mask_array = inference_output['masks'][0,0].mul(255).byte().cpu().numpy()
    mask_array = np.asarray(Image.fromarray(mask_array).resize(image_size))

    output_img = _transform_pred_img(image_as_numpy_array, mask_array)
    buffer = io.BytesIO()
    output_img.save(buffer, format="PNG")
    im_bytes = buffer.getvalue()

    return im_bytes
    
    
def predict_fn(input_data, model):
    """
    Runs prediction on input data using loaded model.
    Args:
        model_dir (input_data): Points to the location of the model artefacts.
        model_dir (object): Points to the location of the model artefacts.

    Returns:
        prediction: Returns the pickled pytorch model file
    """
    
    if input_data is None: return None
    input_data = Image.open(io.BytesIO(input_data)).convert("RGB")
    image_as_tensor, image_as_numpy_array = preprocess(input_data)
    
    model.eval()
    with torch.no_grad():
        inference_output = model([image_as_tensor])
    logger.info(inference_output)
    
    return postprocess(inference_output, image_as_numpy_array)

def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled tensor"""
    if (request_content_type == 'application/json') or (request_content_type == "application/octet-stream"):
        return Image.open(io.BytesIO(input_data)).convert("RGB")
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.
        pass

# def input_fn(request_body, request_content_type):
#     """An input_fn that loads a base64 image"""
#     try:
#         return Image.open(io.BytesIO(input_data)).convert("RGB")
#     except Exception:
#         pass
# #     if request_content_type == 'application/x-image':
# #         return Image.open(io.BytesIO(input_data)).convert("RGB")
# #     else:
# #         # Handle other content-types here or raise an Exception
# #         # if the content type is not supported.
# #         pass

def output_fn(prediction, accept):
    """An output_fn that loads a base64 image"""
    return 'nadote', accept
#     if accept == 'application/json':
#         output = [{"base64_prediction": base64.b64encode(prediction).decode("utf-8")}]
#         return json.dumps(output), accept
#     raise Exception('Requested unsupported ContentType in Accept: ' + accept)



    