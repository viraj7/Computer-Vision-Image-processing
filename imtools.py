import os
from PIL import Image

def get_image_list(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(’.jpg’)]

def im_resize(im,sz):
    pil_im = Image.fromarray(uint8(im))
    return array(pil_im.resize(sz))

def 
