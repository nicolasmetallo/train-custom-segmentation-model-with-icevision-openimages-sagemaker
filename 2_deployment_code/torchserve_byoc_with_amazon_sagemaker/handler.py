# modified from: 
# - https://github.com/aws-samples/amazon-sagemaker-endpoint-deployment-of-fastai-model-with-torchserve
# - https://github.com/pytorch/serve/tree/master/ts/torch_handler

import base64
import io
import logging
import os
import glob
import json
import numpy as np
import torch
import time
import importlib.util
import cv2

from PIL import Image, ImageChops
from torch.autograd import Variable
from torchvision import transforms

logger = logging.getLogger(__name__)

# TO-DO
# -----------------------------------------------------------------------------
# - Have multiple segmentation masks same picture (i.e. two coins)

# helper funcs
# -----------------------------------------------------------------------------

def _list_classes_from_module(module, parent_class=None):
    """
    Parse user defined module to get all model service classes in it.
    :param module:
    :param parent_class:
    :return: List of model service class definitions
    """
    import inspect
    import itertools
    
    # Parsing the module to get all defined classes
    classes = [cls[1] for cls in inspect.getmembers(module, lambda member: inspect.isclass(member) and
                                                    member.__module__ == module.__name__)]
    # filter classes that is subclass of parent_class
    if parent_class is not None:
        return [c for c in classes if issubclass(c, parent_class)]

    return classes

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

from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from typing import *

def _noop_normalize(image: torch.Tensor) -> torch.Tensor:
    return image

def _noop_resize(
    image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]]
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    return image, target

def remove_internal_model_transforms(model: GeneralizedRCNN):
    model.transform.normalize = _noop_normalize
    model.transform.resize = _noop_resize


# main class
# -----------------------------------------------------------------------------

class DIYSegmentation:
    """
    Base default handler to load torchscript or eager mode [state_dict] models
    Also, provides handle method per torch serve custom model specification
    """
    
    def __init__(self):
        self.model = None
        self.mapping = None
        self.map_location = None
        self.device = None
        self.initialised = None
        self.is_model_scripted = None
        
    def initialise(self, context):
        """Initialize function loads the model.pt file and initialized the model object.
        First try to load torchscript else load eager mode state_dict based model.
        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        Raises:
            RuntimeError: Raises the Runtime error when the model.py is missing
        """
        properties = context.system_properties
        self.map_location = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else self.map_location
        )
        logger.info(f"Device on initialisation is: {self.device}")
        self.manifest = context.manifest
        logger.error(self.manifest)

        model_dir = properties.get("model_dir")
        model_pt_path = None
        if "serializedFile" in self.manifest["model"]:
            serialised_file = self.manifest["model"]["serializedFile"]
            model_pt_path = os.path.join(model_dir, serialised_file)
        # model def file
        model_file = self.manifest["model"].get("modelFile", "")

        if model_file:
            logger.debug("Loading eager model")
            self.model = self._load_pickled_model(model_dir, model_file, model_pt_path)
        else:
            logger.debug("Loading torchscript model")
            if not os.path.isfile(model_pt_path):
                raise RuntimeError("Missing the model.pt file")
            self.model = self._load_torchscript_model(model_pt_path)
            self.is_model_scripted = True

        self.model.to(self.device)
        self.model.eval()
        remove_internal_model_transforms(self.model) # remove internal transforms (e.g. resize, normalize)

        logger.debug(f"Model file `{model_pt_path}` loaded successfully")

        self.initialised = True
        
    def _load_torchscript_model(self, model_pt_path):
        """Loads the PyTorch model and returns the NN model object.
        Args:
            model_pt_path (str): denotes the path of the model file.
        Returns:
            (NN Model Object) : Loads the model object.
        """
        return torch.jit.load(model_pt_path, map_location=self.map_location)

    def _load_pickled_model(self, model_dir, model_file, model_pt_path):
        """
        Loads the pickle file from the given model path.
        Args:
            model_dir (str): Points to the location of the model artefacts.
            model_file (.py): the file which contains the model class.
            model_pt_path (str): points to the location of the model pickle file.
        Raises:
            RuntimeError: It raises this error when the model.py file is missing.
            ValueError: Raises value error when there is more than one class in the label,
                        since the mapping supports only one label per class.
        Returns:
            serialized model file: Returns the pickled pytorch model file
        """
        model_def_path = os.path.join(model_dir, model_file)
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model.py file")

        module = importlib.import_module(model_file.split(".")[0])
        model_class_definitions = _list_classes_from_module(module)
        if len(model_class_definitions) != 1:
            raise ValueError(
                "Expected only one class as model definition. {}".format(
                    model_class_definitions
                )
            )

        model_class = model_class_definitions[0]
        model = model_class()
        if model_pt_path:
            state_dict = torch.load(model_pt_path, map_location=self.map_location)
            model.load_state_dict(state_dict)
        return model
        
    def preprocess(self, data):
        """
        Transforms a PIL image for a DL model
        """
        image = data[0].get("data")
        if image is None: image = data[0].get("body") 
            
        image = Image.open(io.BytesIO(image)).convert("RGB")
        image_tfms = _get_transforms()
        
        return image_tfms(image).to(self.device), np.asarray(image)
    
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

        return [{"base64_prediction": base64.b64encode(im_bytes).decode("utf-8")}]

# init handler
# -----------------------------------------------------------------------------

_service = DIYSegmentation()

def handle(data, context):
    if not _service.initialised:
        _service.initialise(context)

    if data is None:
        return None

    data, image_as_numpy_array = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data,image_as_numpy_array)

    return data
