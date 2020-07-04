

Blur persons in photo.

Person detection based on pretrained Deeplabv3 tensorflow model:
https://averdones.github.io/real-time-semantic-image-segmentation-with-deeplab-in-tensorflow/

To run the script you must have tensorflow installed.
For this you could use anaconda to create a tensorflow environment:
    conda env create -f environment.yml
And activate it
    conda activate tensorflow 
Before executing the script:
    ./blur_persons.py  my_image.jpg
