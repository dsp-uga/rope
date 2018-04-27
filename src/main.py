""""
Google Landmark prediction challenge main file

This Challenge has been given a try as part of the 
Spring 2018 Data Science Practicum Class at University of Georgia


Challenge will try to classify images into 15000 classes of landamrks 
as per provided by the training data

@C Team-Rope 2018

this file is partially based on : 
https://github.com/dsp-uga/goucher/blob/master/src/main.py

"""


import argparse
import sys
import os
import logging
from src.preprocessing.downloader import downloader
from src.models.objectClassification import ObjectClassifier

parser = argparse.ArgumentParser(description='Google Landmark Prediction Challenge an Ensemble Way', add_help='How to use', prog='python main.py <options>')

parser.add_argument("-dl", "--download", action='set_true',
                    help='starts download of the files. traincsv, textcsv, traindir, tezstdir should be present')

parser.add_argument("-pp", "--preprocess", action='set_true',
                    help='asks the program to preprocess the data( csv address and output should be supplied! )')

parser.add_argument("-oc", "--classifyobjects", action='store_true',
                    help='add to run the object classification, it will require an input and output directory')

parser.add_argument("-sb", "--storebatch", default=10000 , type=int,
                    help='number of records per numpy storage file')

parser.add_argument("-trcsv", "--traincsv", default='train.csv',
                    help='path to the train.csv file, this will contain training file index.[DEFAULT:train.csv]')

parser.add_argument("-tscsv", "--testcsv", default='test.csv',
                    help='path to the test.csv file, this will contain testing file index.[DEFAULT:test.csv]')

parser.add_argument("-trnpdir", "--trainnpdir", default='../data/train_np/',
                    help='path to the directory which contains training numpy files.[DEFAULT:../data/train_np/]')

parser.add_argument("-tsnpdir", "--testnpdir", default='../data/test_np/',
                    help='path to the directory which contains testing numpy files.[DEFAULT:../data/test_np/]')

parser.add_argument("-trdir", "--traindir", default='../data/train/',
                    help='path to the directory which contains training files.[DEFAULT:../data/train/]')

parser.add_argument("-tsdir", "--testdir", default='../data/test/',
                    help='path to the directory which contains testing files.[DEFAULT:../data/test/]')

parser.add_argument("-lf", "--logfile", default="log.log",
                    help="Path to the log file, this file will contain the log records")

parser.add_argument("-isz", "--imagesize", default="64",
                    help='size of the produced Images [ DEFAULT :64]')

parser.add_argument("-idir", "--inputdir", default=None,
                    help='path to input dir')

parser.add_argument("-odir", "--outputdir", default=None,
                    help='path to output dir')



# compile arguments
args = parser.parse_args()

# setup logging
logging.basicConfig(filename=args.logfile, level=logging.INFO, filemode="w",
                    format=" %(asctime)s - %(module)s.%(funcName)s - %(levelname)s : %(message)s ")


# ensure main folders are there
if not os.path.isdir(args.traindir):
    os.makedirs(args.traindir)

if not os.path.isdir(args.testdir):
    os.makedirs(args.testdir)

# check if preprocess is chosen make sure directory is there
if args.preprocess :

    if not os.path.isdir(args.trainnpdir):
        os.makedirs(args.trainnpdir)

    if not os.path.isdir(args.testnpdir):
        os.makedirs(args.testnpdir)


# download the files if args are set
if args.download:
    if not os.path.isfile(args.traincsv) or not os.path.isfile(args.testcsv):
        print("required files not present, exiting .....!")

    dl = downloader(test_dir=args.testdir, test_file=args.testcsv, train_dir=args.traindir, train_file=args.traincsv)
    logging.info('starting to download files')
    dl.loader()
    logging.info('done downloading files')

# run the classification
if args.classifyobjects:
    logging.info("starting to classify objects")
    cls = ObjectClassifier(input_dir= args.inputdir, output_dir=args.outputdir)
    cls.classify()
    logging.info("done classifying objects")