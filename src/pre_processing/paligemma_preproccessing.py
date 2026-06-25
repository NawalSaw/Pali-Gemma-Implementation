from ast import Tuple
from PIL import Image
from typing import List, Dict, Optional, Union
import torch.nn as nn
import numpy as np
import torch

def proccess_image(
    images: List[Image.Image], 
    size: Dict[str, int] = None, 
    resample: Image.Resampling = None, 
    rescale_factor: float = None, 
    image_mean: Optional[Union[float, List[float]]] = None, 
    image_std: Optional[Union[float, List[float]]] = None
) -> List[np.ndarray]:

    # shape of images: (batch_size, height, width, channels)
    height, width = size[0], size[1]
    images = [resize(image=image, resample=resample, size=(height, width)) for image in images]

    images = [np.array(image) for image in images]
    images = [rescale(image, rescale_factor) for image in images]
    images = [normalize(image, image_mean, image_std) for image in images]

    # shape of images: (batch_size, channels, height, width)
    images = [image.transpose(2, 0, 1) for image in images]

    return images

def add_image_tokens_to_prompt(prompt: str, bos_token: str, image_seq_length: int, image_token: str = IMAGE_TOKEN) -> str:
    return f"{image_token * image_seq_length}{bos_token}{prompt}\n"
    

def resize(image: Image.Image, resample: Image.Resampling, size: Tuple[int, int], reducing_gap: Optional[float] = None) -> Image.Image:
    height, width = image.size
    resized_image = image.resize(
        (height, width), 
        resample=resample,
        reducing_gap=reducing_gap
    )
    return resized_image

def rescale(image: np.ndarray, rescale_factor: float) -> np.ndarray:
    rescaled_image = image * rescale_factor
    return rescaled_image

def normalize(image: np.ndarray, image_mean: Optional[Union[float, List[float]]], image_std: Optional[Union[float, List[float]]]) -> np.ndarray:
    normalized_image = (image - image_mean) / image_std
    return normalized_image

IMAGE_TOKEN = "<image>"
IMAGE_MEAN=[0.5, 0.5, 0.5]
IMAGE_STD=[0.5, 0.5, 0.5]

class PaligemmaPreprocessing(nn.Module):
    def __init__(self, tokenizer, num_image_tokens, image_size):
        """
        Args:
            tokenizer: The tokenizer to use
            num_image_tokens: The number of image tokens
            image_size: The size of the image
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.num_image_tokens = num_image_tokens
        self.image_size = image_size

        token_to_add = {"special_tokens": [IMAGE_TOKEN]}
        tokenizer.add_special_tokens(token_to_add)

        EXTRA_TOKENS = [ f"<loc_{i:04d}>" for i in range(1024)] + [f"<seg_{i:03d}>" for i in range(128)]
        tokenizer.add_tokens(EXTRA_TOKENS)

        self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

    def __call__(self, images: List[Image.Image], texts: List[str], padding: str = "longest", truncation: bool = True):
        assert len(images) == 1 and len(texts) == 1, f"Recieved {len(images)} images and {len(texts)} texts"

        pixel_values = proccess_image(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.BILINEAR, 
            rescale_factor=1/255.0,
            image_mean=IMAGE_MEAN,
            image_std=IMAGE_STD
        )

        pixel_values = np.stack(pixel_values)
        pixel_values = torch.tensor(pixel_values)

        input_strings = [add_image_tokens_to_prompt(
            prompt=prompt,
            bos_token=self.tokenizer.bos_token,
            image_seq_length=self.num_image_tokens,
            image_token=IMAGE_TOKEN,
        ) for prompt in texts]
        
        inputs = self.tokenizer(input_strings, return_tensors="pt", padding=padding, truncation=truncation)
        
        return_data = {
            "pixel_values": pixel_values,
            **inputs
        }

        return return_data