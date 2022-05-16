import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
import sklearn.cluster
from opts import get_opts

def extract_filter_responses(opts, img):



    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)

    [hint]
    * To produce the expected collage, loop first over scales, then filter types, then color channel.
    * Note the order argument when using scipy.ndimage.gaussian_filter. 
    '''
    

    # ----- TODO -----
    img = np.tile(img[:, :, np.newaxis], (1,1,3)) if len(img.shape) == 2 else img
    img = img[:, :, 0:3] if img.shape[2] > 3 else img
    lab_image = skimage.color.rgb2lab(img)
    x,l=12,0
    filter_responses = np.empty((img.shape[0], img.shape[1], x*len(opts.filter_scales)))
    for k in range(len(opts.filter_scales)):

            for i in range(3):#created gaussian filter
                    filter_responses[:, :, k*x+i] = scipy.ndimage.gaussian_filter(lab_image[:, :, i], opts.filter_scales[k])
                    #created laplace
                    filter_responses[:, :, k*x+3*(l+1)+i] = scipy.ndimage.gaussian_laplace(lab_image[:, :, i], opts.filter_scales[k])
                    #created  gaussian derivative in x
                    filter_responses[:, :, k*x+3*(l+2)+i] = scipy.ndimage.gaussian_filter(lab_image[:, :, i], opts.filter_scales[k], [0, 1])
                    #created gaussian derivative in y
                    filter_responses[:, :, k*x+3*(l+3)+i] = scipy.ndimage.gaussian_filter(lab_image[:, :, i], opts.filter_scales[k], [1, 0])
    return filter_responses #return response of filter



def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''

    # ----- TODO -----
    i, alpha, path = args
    img = Image.open('../data/'+(path))
    img = np.array(img).astype(np.float32) / 255
    ap=int(alpha)
    opts=get_opts()

    filter_response = extract_filter_responses(opts, img)
    #collecting response of filter at random
    x0= np.random.choice(filter_response.shape[0], ap)
    x1 = np.random.choice(filter_response.shape[1], ap)

    subI = filter_response[x0, x1, :]
    #saving response in .npy format
    np.save(os.path.join('../temp/', str(i) + '.npy'), subI)
def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    #  For testing purpose, you can create a train_files_small.txt to only load a few images.
    train_files1 = open(join(data_dir, 'train_files.txt')).read().splitlines()
    # ----- TODO -----
    size1 = len(train_files1)
    #creating a zipe file to perfrom multiprocessing
    args = list(zip(np.arange(size1), np.ones(size1) * opts.alpha, train_files1))
    multiprocessing.Pool(n_worker).map(compute_dictionary_one_image, args)

    filter_responses =[]
    for i in range(len(train_files1)):
        temp_files = np.load('../temp/' + str(i)+'.npy')
        filter_responses.append(temp_files)#adding up the temp file  to fliter

    filter_responses = np.concatenate(filter_responses, axis=0)
    # performing kmean cluster form the response
    kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(filter_responses)
    dictionary = kmeans.cluster_centers_
    #saving collected data in form of dictionary

    np.save(join(out_dir, 'dictionary.npy'), dictionary)

    ## example code snippet to save the dictionary
    # np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    # ----- TODO -----
    filter_response = extract_filter_responses(opts,img).reshape(img.shape[0] * img.shape[1], -1)
    dist1 = scipy.spatial.distance.cdist(filter_response, dictionary,'euclidean')
    wordmap = np.argmin(dist1, axis = 1).reshape(img.shape[0], img.shape[1])
    return wordmap
