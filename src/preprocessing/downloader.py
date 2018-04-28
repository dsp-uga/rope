# -*- coding: utf-8 -*-

# !/usr/bin/python

# Note: requires the tqdm package (pip install tqdm)

# Note to Kagglers: This script will not run directly in Kaggle kernels. You
# need to download it and run it on your local machine.

# Downloads images from the Google Landmarks dataset using multiple threads.
# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.

# this code is based on a kaggle kernel at :
# https://www.kaggle.com/anokas/python3-dataset-downloader-with-progress-bar

import os, multiprocessing, csv
from urllib import request
from PIL import Image
from io import BytesIO
import logging

class downloader():
    """
    This class handles downloading of the test and train samples
    
    """

    def __init__(self , train_dir , test_dir , train_file, test_file ):
        self.train_file = train_file
        self.test_file = test_file
        self.train_dir = train_dir
        self.test_dir = test_dir


    def parse_data(self, data_file):
        csvfile = open(data_file, 'r')
        csvreader = csv.reader(csvfile)
        key_url_list = [line[:2] for line in csvreader]
        return key_url_list[1:]  # Chop off header


    def download_image(self, key_url , out_dir ):
        """
        downloads an image 
        :param key_url: URL to download from 
        :param out_dir: place to save to
        :return: 
        """

        (key, url) = key_url
        filename = os.path.join(out_dir, '{}.jpg'.format(key))

        if os.path.exists(filename):
            print('Image {} already exists. Skipping download.'.format(filename))
            return 0

        try:
            response = request.urlopen(url)
            image_data = response.read()
        except:
            print('Warning: Could not download image {} from {}'.format(key, url))
            return 1

        try:
            pil_image = Image.open(BytesIO(image_data))
        except:
            print('Warning: Failed to parse image {}'.format(key))
            return 1

        try:
            pil_image_rgb = pil_image.convert('RGB')
        except:
            print('Warning: Failed to convert image {} to RGB'.format(key))
            return 1

        try:
            pil_image_rgb.save(filename, format='JPEG', quality=90)
        except:
            print('Warning: Failed to save image {}'.format(filename))
            return 1

        return 0

    def download_image_train(self , key_url):
        return self.download_image(key_url, self.train_dir)

    def download_image_test(self , key_url):
        return self.download_image(key_url, self.test_dir)


    def loader(self):
        """
        strts the download of test and train images
        :return: 
        """
        key_url_list = self.parse_data(self.train_file)
        pool = multiprocessing.Pool(processes=20)  # Num of CPUs
        failures = sum(tqdm.tqdm(pool.imap_unordered(self.download_image_train, key_url_list), total=len(key_url_list)))
        print('Total number of download failures:', failures)
        logging.info("Total number of download failures in training: %d " % failures)

        pool.close()
        pool.terminate()

        key_url_list = self.parse_data(self.test_file)
        pool = multiprocessing.Pool(processes=20)  # Num of CPUs
        failures = sum(tqdm.tqdm(pool.imap_unordered(self.download_image_test, key_url_list ), total=len(key_url_list)))
        print('Total number of download failures:', failures)
        logging.info("Total number of download failures in testing: %d " % failures)
        pool.close()
        pool.terminate()