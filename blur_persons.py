#!/usr/bin/env python3

"""
Blur persons in photo.

Person detection based on pretrained Deeplabv3 tensorflow model:
https://averdones.github.io/real-time-semantic-image-segmentation-with-deeplab-in-tensorflow/
"""

import os
import io
import sys
import math
import glob
import urllib.request
import tarfile
import os.path
import argparse
import platform
import tempfile
import subprocess
import collections

import numpy as np

from PIL import Image, ImageDraw, ImageFilter, ImageColor

from packaging import version

try:
    import tensorflow.compat.v1 as tf
    if version.parse(tf.__version__) < version.parse('1.5'):
        raise ImportError('Please upgrade your tensorflow installation to v1.5.0 or newer!')
except:
    print('tensorflow.compat.v1 not available')

try:
    import tflite_runtime.interpreter as tflite
except:
    print('tflite_runtime not available')


COCO_LABEL_NAMES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
    'train', 'tv'
]


CITYSCAPE_LABEL_NAMES = [
    'unlabeled',
    'ego vehicle',
    'rectification border',
    'out of roi',
    'static',
    'dynamic',
    'ground',
    'road',
    'sidewalk',
    'parking',
    'rail track',
    'building',
    'wall',
    'fence',
    'guard rail',
    'bridge',
    'tunnel',
    'pole',
    'polegroup',
    'traffic light',
    'traffic sign',
    'vegetation',
    'terrain',
    'sky',
    'person',
    'rider',
    'car',
    'truck',
    'bus',
    'caravan',
    'trailer',
    'train',
    'motorcycle',
    'bicycle',
]

ModelConfig = collections.namedtuple('ModelConfig', ['name', 'url', 'label_names'])

MODEL_CONFIGS = {name : ModelConfig(name, url, label_names) for name, url, label_names in [
    ('xception_coco_voctrainaug',    'http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz', COCO_LABEL_NAMES),
    ('xception_coco_voctrainval',  'http://download.tensorflow.org/models/deeplabv3_pascal_trainval_2018_01_04.tar.gz',  COCO_LABEL_NAMES),
    ('xception_cityscapes_trainfine','http://download.tensorflow.org/models/deeplabv3_cityscapes_train_2018_02_06.tar.gz', CITYSCAPE_LABEL_NAMES),
]}

class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    FROZEN_GRAPH_NAME = 'frozen_inference_graph'
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, resized_image):
        """Runs inference on a single image.

        Args:
            image: A PIL.Image object, raw input image.

        Returns:
            resized_image: RGB image resized from original input image.
            segmentation_map: Segmentation map of `resized_image`.
        """
        batch_segmentation_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        segmentation_map = batch_segmentation_map[0]
        return resized_image, segmentation_map


class LiteModel(object):
    """Class to load TF Lite model and run inference."""

    EDGETPU_SHARED_LIB = {
      'Linux': 'libedgetpu.so.1',
      'Darwin': 'libedgetpu.1.dylib',
      'Windows': 'edgetpu.dll'
    }[platform.system()]

    def __init__(self, tflite_path):
        device = None
        self.interpreter = tflite.Interpreter(
            model_path=tflite_path,
            experimental_delegates=[
                tflite.load_delegate(self.EDGETPU_SHARED_LIB, {'device': device[0]} if device else {})
            ]
        )

        # Load the TFLite model and allocate tensors.
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()

        self.input_size = self.input_details['shape'][2], self.input_details['shape'][1]
        self.INPUT_SIZE = self.input_size[0]
        print(f'Model input size {self.input_size}')

    def run(self, resized_image):
        """Runs inference on a single image.

        Args:
            image: A PIL.Image object, raw input image.

        Returns:
            resized_image: RGB image resized from original input image.
            segmentation_map: Segmentation map of `resized_image`.
        """
        if resized_image.size == self.input_size:
            analyzed_image = resized_image
        else:
            analyzed_image = Image.new('RGB', self.input_size, color='black')
            analyzed_image.paste(resized_image)

        input_data = np.asarray(analyzed_image)
        input_data = np.expand_dims(input_data, 0)

        self.interpreter.set_tensor(self.input_details['index'], input_data)
        self.interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        segmentation_map = output_data[0]

        if resized_image.size != self.input_size:
            segmentation_map = segmentation_map[0:resized_image.size[0], 0:resized_image.size[1]]

        return resized_image, segmentation_map

def get_new_filename(filename, suffix, dest):
    new_filename = filename
    if suffix is not None:
        base, ext = os.path.splitext(new_filename)
        new_filename = base + suffix + ext
    if dest is not None:
        new_filename = os.path.join(dest, os.path.basename(new_filename))
    return new_filename

def save_and_copy_exif(image, source_path, save_path, *args, **kwargs):
    base, ext = os.path.splitext(save_path)
    tmp_path = base + ".tmp" + ext
    image.save(tmp_path, *args, **kwargs)
    subprocess.check_call([
        "exiftool", "-overwrite_original", "-TagsFromFile",
         source_path, tmp_path])
    os.rename(tmp_path, save_path)

def split_area(area_width, area_height, box_width, box_height, is_360=None, overlap_factor=0.15):
    """
    Yield a list of box (x1,y1,x2,y2)
    representing cut of the areas in
    overlapping smaller pieces.

    If is_360 is True, the returned last boxes will be larger than the area
    in order to wrap back to the left of the area.
    """
    overlap_factor = max(0.0, min(0.5, overlap_factor)) # ensure in range 0.0 … 0.5
    if is_360 is None:
        is_360 = (area_width == (2 * area_height))
    real_area_width = (area_width + box_width*overlap_factor) if is_360 else area_width
    nb_x = math.ceil(real_area_width / (box_width - overlap_factor*box_width)) if real_area_width > box_width else 1
    while (nb_x > 2) and ((box_width*(1.0-overlap_factor)*(nb_x-2) + box_width) > real_area_width):
        nb_x = nb_x - 1
    nb_y = math.ceil(area_height / (box_height - overlap_factor*box_height)) if area_height > box_height else 1
    while (nb_y > 2) and ((box_height*(1.0-overlap_factor)*(nb_y-2) + box_height) > area_height):
        nb_y = nb_y - 1
    if nb_x > 1:
        factor_width = (real_area_width - box_width) / (box_width * (nb_x - 1))
    else:
        factor_width = 0.0
    if nb_y > 1:
        factor_height = (area_height - box_height) / (box_height * (nb_y - 1))
    else:
        factor_height = 0.0
    for i in range(nb_x):
        x = int(i*box_width*factor_width)
        for j in range(nb_y):
            y = int(j*box_height*factor_height)
            yield(x, y, x+box_width, y+box_height)

def iter_image_sub_boxes(image_width, image_height, box_size, is_360=None, overlap_factor=0.15):
    box_width = min(box_size, image_width)
    box_height = min(box_size, image_height)
    for result in split_area(image_width, image_height, box_width, box_height, is_360, overlap_factor):
        yield result
    if min(image_width, image_height) >= 2*box_size:
        box_size = min(image_width, image_height)
        for result in split_area(image_width, image_height, box_size, box_size, is_360, overlap_factor):
            yield result

def blur_from_model_and_colormap(original_image, model, colormap, blur, dezoom=1.0, mask=False):
    width, height = original_image.size
    is_360 = (width == (2 * height))
    if type(blur) is int:
        blurred_im = original_image.filter(ImageFilter.GaussianBlur(radius=blur))
    else:
        blurred_im = Image.new('RGB', original_image.size, color=blur)
    if mask:
        new_image = Image.new('RGB', original_image.size, color='black')
    else:
        new_image = original_image.copy()
    for x1,y1,x2,y2 in iter_image_sub_boxes(width, height, int(model.INPUT_SIZE*dezoom)):
        extract_width = x2-x1
        extract_height = y2-y1
        print("search in box", (x1,y1,x2,y2), "size %dx%d" % (extract_width , extract_height))
        if x2 >= width and is_360:
            # The (x1,y1,x2,y2) box wrap from the far right of the image back to
            # the left (360° image), so we need to take two boxes, one on the
            # right and one on the left:
            x2_1 = width
            x2_2 = x2 - width
            extract_width_1 = x2_1 - x1
            extract_width_2 = x2_2
            extract_image = Image.new('RGB', (extract_width, extract_height))
            extract_image.paste(original_image.crop((x1,y1, x2_1, y2)), (0,0))
            extract_image.paste(original_image.crop((0,y1, x2_2, y2)), (extract_width_1,0))
        else:
            extract_image = original_image.crop((x1,y1, x2,y2))

        extract_width, extract_height = extract_image.size
        resize_ratio = 1.0 * model.INPUT_SIZE / max(extract_width, extract_height)
        target_size = (int(resize_ratio * extract_width), int(resize_ratio * extract_height))
        resized_image = extract_image.convert('RGB').resize(target_size, Image.ANTIALIAS)

        resized_im, segmentation_map = model.run(resized_image)
        segmentation_mask = Image.fromarray(np.uint8(colormap[segmentation_map])).resize((x2-x1, y2-y1), Image.NEAREST if mask else Image.ANTIALIAS)
        if x2 >= width and is_360:
            new_image.paste(blurred_im.crop((x1,y1, x2_1,y2)), (x1,y1), segmentation_mask.crop((0,0, extract_width_1, extract_height)))
            new_image.paste(blurred_im.crop((0,y1, x2_2,y2)), (0,y1), segmentation_mask.crop((extract_width_1,0, extract_width, extract_height)))
        else:
            new_image.paste(blurred_im.crop((x1,y1, x2,y2)), (x1,y1), segmentation_mask)
    if mask:
        new_image = new_image.convert("1")
    return new_image

def get_image_quality(image_path, default=None):
    try:
        for line in subprocess.check_output(["identify", "-verbose", image_path]).decode().splitlines():
            if line.strip().lower().startswith("quality:"):
                return int(line.split(":")[1])
    except:
        pass
    return default



def blur_in_files(files, model, classes, blur, dest, suffix, dezoom, quality, mask, lite):
    config = MODEL_CONFIGS[model]
    if lite:
        # https://coral.ai/models/
        url = 'https://github.com/google-coral/edgetpu/raw/master/test_data/deeplabv3_mnv2_pascal_quant_edgetpu.tflite'
        download_path = 'deeplabv3_mnv2_pascal_quant_edgetpu.tflite'
        if not os.path.exists(download_path):
            print('downloading model to %s, this might take a while...' % download_path)
            urllib.request.urlretrieve(url, download_path + ".tmp")
            os.rename(download_path + ".tmp", download_path)
            print('download completed!')

        model = LiteModel(download_path)
    else:
        tarball_name = os.path.basename(config.url)
        model_dir = os.path.join(os.path.dirname(__file__), config.name) # or tempfile.mkdtemp()
        download_path = os.path.join(model_dir, tarball_name)
        if not os.path.exists(download_path):
            tf.gfile.MakeDirs(model_dir)
            print('downloading model to %s, this might take a while...' % download_path)
            urllib.request.urlretrieve(config.url, download_path + ".tmp")
            os.rename(download_path + ".tmp", download_path)
            print('download completed!')
        print("load model", download_path)

        model = DeepLabModel(download_path)

    blur_colormap = np.zeros((512,4), dtype=int)
    for clazz in classes:
        index = config.label_names.index(clazz)
        blur_colormap[index] = (255,255,255,255)

    for filename in files:
        new_filename = get_new_filename(filename, suffix, dest)
        if mask:
            new_filename = new_filename.rsplit('.', 1)[0] + '.png'
        print(filename, "->", new_filename)
        original_image = Image.open(filename)
        new_image = blur_from_model_and_colormap(original_image, model, blur_colormap, blur, dezoom, mask)
        if mask:
            new_image.save(new_filename)
        else:
            this_quality = get_image_quality(filename, "maximum") if quality is None else quality
            save_and_copy_exif(new_image, filename, new_filename, quality=this_quality)

def check_dir(name):
    assert os.path.isdir(name)
    return name

def may_be_int(value):
    try:
        return int(value)
    except:
        pass
    return value

def int_or_color(value):
    try:
        return int(value)
    except:
        pass
    return ImageColor.getrgb(value)

def main(args):
    model="xception_coco_voctrainval"
    config=MODEL_CONFIGS[model]
    parser = argparse.ArgumentParser(
        description="Blur persons from photos.")
    parser.add_argument("-s", "--suffix", default=None,
        help="suffix for modified image filename")
    parser.add_argument("-d", "--dest", type=check_dir, default=None,
        help="destination directory for modiffied image")
    parser.add_argument("-z", "--dezoom", type=float, default=1.0,
        help="dezoom factor (e.g. 2.0) for faster search in smaller image (default=1 for search at original resolution)")
    parser.add_argument("-q", "--quality", type=may_be_int,
        help="quality option of saved images (e.g. 75 or maximum)")
    parser.add_argument("-b", "--blur", default=30, type=int_or_color,
        help="blur radius in pixel, or a flat color name or #RGB")
    parser.add_argument("-c", "--class", action="append",
        choices=config.label_names,
        help="add a class of items to blur (the default is 'person' if no class is specified)")
    parser.add_argument("-m", "--mask", action="store_true",
        help="Save the mask inside of blur")
    parser.add_argument("-l", "--lite", action="store_true",
        help="Use Tensorflow Lite in place of Tensorflow.")
    parser.add_argument("input", nargs="+")
    options = parser.parse_args(args[1:])
    if options.dest is None and options.suffix is None:
        options.suffix = "-mask" if options.mask else "-blurred"
    classes = getattr(options, "class")
    if classes is None:
        classes = ["person"]
    if options.mask:
        options.blur = 'white'
    blur_in_files(
        files=options.input,
        model=model,
        classes=classes,
        blur=options.blur,
        dest=options.dest,
        suffix=options.suffix,
        dezoom=options.dezoom,
        quality=options.quality,
        mask=options.mask,
        lite=options.lite,
    )

if __name__ == '__main__':
    main(sys.argv)

