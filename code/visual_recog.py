import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import visual_words
from opts import get_opts
import itertools


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    # ----- TODO -----
    out_dir = opts.out_dir
    dictionary = np.load(join(out_dir, 'dictionary.npy'))
    dict_size = len(dictionary)
    #using numpy to produce  histogram of a image

    return np.histogram(wordmap, bins=dict_size, density=True)

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K
    L = opts.L
    # ----- TODO -----
    out_dir = opts.out_dir
    dictionary = np.load(join(out_dir, 'dictionary.npy'))
    dict_size = len(dictionary)
    hist = np.array([])
    #creating array of histogram list
    hists_list =list(np.zeros(L))
    hists_list[-1] = sub_hist(wordmap)
    for l in range(L - 2, -1, -1):

            cnum = 2 ** l
            sub_hists = np.zeros((cnum, cnum, dict_size))
            for i in range(cnum):
                        for j in range(cnum):
                                 sub_hists[i][j] = hists_list[l + 1][i * 2: (i + 1) * 2, j * 2: (j + 1) * 2].reshape(4, -1).sum(axis = 0)
            hists_list[l] = sub_hists

    for l in range(L):
            if l in [0, 1]:#assign weight of layer with two different condition
                    weight = 2.0 ** (-L + 1)
            else:
                    weight = 2.0 ** (l - L)
            sub_hists = hists_list[l]
            for i in range(len(sub_hists)):
                    for j in range(len(sub_hists[i])):
                            hist = np.concatenate((hist, sub_hists[i][j] * weight))
    hist = hist / hist.sum()# calculating histogram
    return hist
    
def get_image_feature(args):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
    opts = get_opts()
    data_dir =  opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L
    dictionary = np.load(join(out_dir, 'dictionary.npy'))
    dict_size = len(dictionary)
    labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)

    # ----- TODO -----
    #extracting args file to assign image path
    i, img_path = args
    img = Image.open( opts.data_dir+"/"+(img_path))
    img = np.array(img).astype(np.float32)/255
    #performing wordmap
    wordmap = visual_words.get_visual_words(opts, img, dictionary)

       #extracting feature  from wordmap
    features = get_feature_from_wordmap_SPM(opts, wordmap)

     # stori the trained file for temparory
    np.savez("../temp/" + "train_"+str(i)+".npz", features=features, labels=labels, allow_pickle=True)

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    # ----- TODO -----
    img_list = (np.arange(len(train_files)))
    args = list(itertools.zip_longest(img_list, train_files))
    #constructing zip file and performing multiprocessing to get feature of image
    multiprocessing.Pool(n_worker).map(get_image_feature,  args)
    features = []#delclare  list
    for i in range(len(train_files)):
        temp = np.load('../temp/'+'train_'+str(i)+'.npz', allow_pickle=True)
        feature1 = temp['features']
        features.append(feature1)#loading feature of image
        #compressing and storig trained file in the system

    np.savez_compressed(join(out_dir, 'trained_system.npz'),features=features,labels=train_labels,dictionary=dictionary,SPM_layer_num=SPM_layer_num,)
    ## example code snippet to save the learned system
    # np.savez_compressed(join(out_dir, 'trained_system.npz'),
    #     features=features,
    #     labels=train_labels,
    #     dictionary=dictionary,
    #     SPM_layer_num=SPM_layer_num,
    # )

def similarity_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    hist_minimum = np.minimum(word_hist, histograms)#extracting minimun bin value
    similarity = np.sum( hist_minimum, axis=1)#performing sum to find similarity

    return similarity#return value

def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)
    features = trained_system['features']
    train_labels = trained_system['labels']
    # ----- TODO -----

    # constructing 8*8 matrices of zeros
    conf = np.zeros((8, 8))

    for i in range(len(test_files)):
        img = Image.open(data_dir + '/' + (test_files[i]))
        img = np.array(img).astype(np.float32)/255
        #extracting wordmap
        wordmap = visual_words.get_visual_words(opts, img, dictionary)
        #gettting feature of wordmap
        test1= get_feature_from_wordmap_SPM(opts, wordmap)
        #performing similarity for features and pridicting label how much similar
        predict_label = train_labels[np.argmax(similarity_to_set(test1, features))]
        #collecting  actual labels
        actual_label = test_labels[i]
        #creating confusion matrix
        conf[actual_label, predict_label] += 1
#finding accuracy from actual and predicted matrix
    accuracy = np.trace(conf) / np.sum(conf)


    return conf, accuracy

def sub_hist(wordmap):
        opts=get_opts()
        L = opts.L
        out_dir = opts.out_dir
        dictionary = np.load(join(out_dir, 'dictionary.npy'))
        dict_size = len(dictionary)
        ht = wordmap.shape[0] // (2 ** (L - 1))
        wd=wordmap.shape[1] // (2 ** (L - 1))
        hists_list =list(np.zeros(L))
        sub_hists = np.zeros((2 ** (L - 1), 2 ** (L - 1), dict_size))

        for i in range( 2 ** (L - 1)):
               for j in range( 2 ** (L - 1)):
                       sub_fig = wordmap[i * ht:(i + 1) * ht, j * wd:(j + 1) * wd]
                       sub_hist, _ = np.histogram(sub_fig.reshape(-1), bins = range(dict_size + 1))
                       sub_hists[i][j] = sub_hist
        return sub_hists




