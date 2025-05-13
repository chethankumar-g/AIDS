import numpy as np
from PIL import Image

def preprocess_image(image_file) :
    image = Image.open(image_file).convert("RGB")
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0            
    image_array = np.expand_dims(image_array, axis=0) 
    return image_array