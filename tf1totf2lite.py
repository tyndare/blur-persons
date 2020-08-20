import os
import sys
import math
import tarfile
import argparse
import numpy as np
from PIL import Image

import tensorflow.compat.v1 as tf

from blur_persons import iter_image_sub_boxes

INPUT_SIZE = 513


def blur_from_model_and_colormap(filename, input_size, dezoom):
    original_image = Image.open(filename)

    width, height = original_image.size
    is_360 = (width == (2 * height))
    for x1,y1,x2,y2 in iter_image_sub_boxes(width, height, int(input_size*dezoom)):
        extract_width = x2-x1
        extract_height = y2-y1
        print("search in box", (x1,y1,x2,y2), "size %dx%d" % (extract_width , extract_height))
        if x2 >= width and is_360:
            # The (x1,y1,x2,y2) box wrap from the far right of the image back to
            # the left (360° image), so we need to take to boxes, one on the
            # right, on on the left:
            x2_1 = width
            x2_2 = x2 - width
            extract_width_1 = x2_1 - x1
            extract_width_2 = x2_2
            extract_image = Image.new('RGB', (extract_width, extract_height))
            extract_image.paste(original_image.crop((x1,y1, x2_1, y2)), (0,0))
            extract_image.paste(original_image.crop((0,y1, x2_2, y2)), (extract_width_1,0))
        else:
            extract_image = original_image.crop((x1,y1, x2,y2))

        width, height = extract_image.size
        resize_ratio = 1.0 * input_size / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = extract_image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        if target_size == (input_size, input_size):
            analyzed_image = resized_image
        else:
            analyzed_image = Image.new('RGB', (input_size, input_size), color='black')
            analyzed_image.paste(resized_image)

        input_data = np.asarray(analyzed_image, dtype=np.float32)

        input_data = np.expand_dims(input_data, 0)
        input_data = input_data / 127.5 - 1

        yield input_data


def representative_dataset_gen():
    list_of_files = os.listdir('representative_dataset')
    for i, path in enumerate(sorted(list_of_files)):
        print(f'Quantize with representative data {i}/{len(list_of_files)}')
        for ib in blur_from_model_and_colormap('representative_dataset/' + path, INPUT_SIZE, 6):
            yield [ib]


def convert(input, output, quantization):
    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        input,
        input_arrays = ['sub_7'],
        output_arrays = ['ResizeBilinear_2'],
        # input_shapes = {'ImageTensor': [1, INPUT_SIZE, INPUT_SIZE, 3]}
    )

    # https://www.tensorflow.org/lite/performance/post_training_quantization
    if quantization in ('dr', 'dynamic_range'):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    elif quantization in ('f16', 'float16'):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    elif quantization in ('ui8', 'uint8', 'Fui8', 'full_uint8'):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen

        if quantization in ('Fui8', 'full_uint8'):
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8  # or tf.uint8
            converter.inference_output_type = tf.uint8  # or tf.uint8

    tflite_model = converter.convert()
    open(output, 'wb').write(tflite_model)


def main(args):
    parser = argparse.ArgumentParser(
        description=f'''Convert TF1 model to TF2 Lite.
Example:

{sys.argv[0]} frozen_inference_graph.pb xception_coco_voctrainval.tflite

{sys.argv[0]} -q full_uint8 frozen_inference_graph.pb xception_coco_voctrainval-full_uint8.tflite
''')
    parser.add_argument('-q', '--quantization',
        choices=['dr', 'dynamic_range', 'ui8', 'uint8', 'Fui8', 'full_uint8', 'f16', 'float16'],
        help='''Optimization Quantization
 - dr, dynamic_range: 4x smaller, 2x-3x speedup - CPU (mono-thread)
 - f16, float16: 2x smaller, GPU acceleration - CPU, GPU
 - i8, int8 - Edge TPU, Microcontrollers
 - Fi8, full_int8, 4x smaller, 3x+ speedup - Edge TPU, Microcontrollers

 Quantisation to int require images in representative_dataset.
''')
    parser.add_argument('input')
    parser.add_argument('output')
    options = parser.parse_args(args[1:])
    convert(
        options.input,
        options.output,
        quantization=options.quantization,
    )

if __name__ == '__main__':
    main(sys.argv)
