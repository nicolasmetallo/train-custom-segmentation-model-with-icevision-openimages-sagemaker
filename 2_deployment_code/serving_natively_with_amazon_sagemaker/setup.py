from setuptools import setup, find_packages

setup(name='maskrcnn-torchserve-example',
      version='1.0',
      description='SageMaker Example for Image Segmentation (Mask RCNN).',
      packages=find_packages(exclude=('tests', 'docs')))