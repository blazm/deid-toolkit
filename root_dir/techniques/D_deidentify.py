
from B_k_same_net import clusterImagesEqualGroups, clusterImagesDummy, loadClusteredImages, loadClusters, saveClusters
from B_k_same_net import filterProxy, extractProxyIDs, makeSquare
from A_pick_samples import loadImages, reset_directory, read_files
from A_emo_pick_samples import readEmotions

from os.path import join, exists, split, isfile, isdir


import numpy as np
from scipy.misc import imresize, imsave
from scipy.ndimage.filters import gaussian_filter

# weak pixelization size 8 or 16
def pixelize(images, subs_size=32): # subs_size is fixed, does not change with k (plot straight line!)

    deidentified = []
    
    for i in images:
        h, w, ch = i.shape
        di = i[::subs_size, ::subs_size, :] # subsample by step
        di = imresize(di, (h, w, ch), interp='nearest') # resize to original size
        deidentified.append(di)
    
    return deidentified

# weak blur size_ 15
def blur(images, kernel_size=30): # kernel_size is fixed, does not change with k (plot straight line!)
    
    sigma = kernel_size / 2 # TODO: review this
    deidentified = []
    
    # filter size is: https://stackoverflow.com/questions/25216382/gaussian-filter-in-scipy 
    # w = 2*int(truncate*sigma + 0.5) + 1
    # (w - 1)/2 = int(truncate*sigma + 0.5)
    # For w = 5, the left side is 2. The right side is 2 if
    # 2 <= truncate*sigma + 0.5 < 3

    for i in images:
        di = gaussian_filter(i, (sigma, sigma, 0))
        deidentified.append(di)
    
    return deidentified
    
def k_same_pixel(clusters, clustered_images, k=2):
    
    deidentified = {}

    for k in clusters.keys():
        imgs = np.array(clustered_images[k])
        di = np.mean(imgs, axis=0) # check if axis ok 
        deidentified[k] = di
    
    return deidentified

def k_same_M(clusters, clustered_images, k): # AAM
    
    # images preprocessed to AAM canonical form using MATLAB
    # frontalization, landmark detection, delaunay triangulation
    images = k_same_pixel(clusters, clustered_images, k)

    ''' TODO: 
        cluster images
        find landmarks
        align to canonic landmark form
        (texture - triangles + landmark coordinates)
        perform averaging on aligned canonic images
    '''

    return images 

'''
def remapNamesToIds(cluster_files):
    ids = {}
    for k in cluster_files.keys():
        for item in cluster_files[k]:
            iden = (int)(item.split("_")[0][3:])            
            if k in ids:
                ids[k].append(iden)
            else:
                ids[k] = [iden]
    return ids
'''

def k_same_net(clustered_probes, clustered_images, k=2):
    
    from keras import backend as K
    K.set_image_dim_ordering('tf')
    from Generator import Generator

    #do_emotions = False

    deconv_layer = 6 # 5 or 6
    model_name = 'FaceGen.RaFD.model.d{}.adam'.format(deconv_layer)
#    model_path = '../de-id/generator/output/{}.h5'.format(model_name)
    model_path = './models/{}.h5'.format(model_name) # locally stored models

    gen = Generator(model_path, deconv_layer=deconv_layer)

    proxy_path = "../de-id/DB/rafd2-frontal/"
    proxys = read_files(proxy_path)
    proxys = filterProxy(proxys)
    #proxys = range(len(proxys))
    #print proxys
    #print(sorted(extractProxyIDs(proxys)))

    # TODO: load from cache
    # if exists load, else generate and display notification
    cluster_file = join(proxy_path, "../", "clusters_k=%d.txt" %(k))
    if not isfile(cluster_file):
        #print proxys
        print "k-same-net: Clustered proxys not available for k=%d, performing image clustering..." %(k)
        clustered_proxys = clusterImagesEqualGroups(proxy_path, proxys, k, 320)
        saveClusters(clustered_proxys, cluster_file)
    else:
        clustered_proxys = loadClusters(cluster_file)

    #clustered_proxys = clusterImagesDummy(proxys, k)
    clustered_proxys = extractProxyIDs(clustered_proxys, proxys)

    ''' TODO: 
        rewrite B_k_same_net.py into function. Then test all these deid implementations in the same way in the protocol
    '''
    #probe_path = os.path.join(fold_path, 'fold_probes_{}/'.format(fold+1))
    #probes = read_files(probe_path) 
    #clustered_probes = clusterImagesDummy(probes, k)
    #print "PROBES:_ ", clustered_probes
    deidentified_images = {}

    #print "CLUSTERED PROBES LEN:", len(clustered_probes)
    # TODO
    #mapIndices = mapRandomly(clustered_probes, clustered_proxys)

    # TODO: map probe and proxy clusters
    # connect clustered probes with clustered proxy set?


    #print clustered_proxys
    for i, probe_cluster in enumerate(clustered_probes):

        probe_cluster = clustered_probes[i]

        #print clustered_proxys

        proxy_cluster = clustered_proxys[i] 

        #print probe_cluster
        #print proxy_cluster

        #print "PROBE CLUSTER LEN:", len(probe_cluster)    
        #print proxy_cluster
        # TODO: estimate emotion from original image
        #emotion = 
        emotion = 'neutral' # 'happy' #'neutral'

        # generate deidentified 
        image = gen.generate(proxy_cluster, emotion)

        # crop and resize img
        # make it square, crop from center
        image = makeSquare(image, newdim=320)
        
        # save generated result for all ids
        
        for item in probe_cluster:

            if do_emotions:
                # TODO: deidentify using emotions for each item
                #print item # read emotion from probe names
                #emo_id = int(item.split('_')[1])
                emo_key = '_'.join(item.split('.')[0].split('_')[:-1])
                
                if emo_key not in emo_gtru:
                    print emo_key, " not in emotion GT!"

                #print emo_gtru
                #emo_id = emo_gtru[emo_key]

                try:
                    #emotion = emotions[emo_id-1] #'happy' # 
                    emotion = item.split('.')[0].split('_')[-1]
                    image = gen.generate(proxy_cluster, emotion)
                except:
                    print item, " has unknown emotion, resetting to 'neutral'!"
                    emotion = 'neutral'
                    image = gen.generate(proxy_cluster, emotion)

                #image = gen.generate(proxy_cluster, emotion)
                image = makeSquare(image, newdim=320)

            #try:
            #    _id = int(item.split('_')[0])
            #except ValueError:
                # handle CK+ DB: ValueError: invalid literal for int() with base 10: 'S053'
            #    _id = int(item.split('_')[0][1:])


            #deidentified_images[i] = image

            # TODO: enable emotions
            if i in deidentified_images:
                deidentified_images[i].append(image)
            else:
                deidentified_images[i] = [image]

            #deidentified_images[probe_cluster] = image

            #_id = int(item.split('_')[0])
        #    path = './fold_probes_{}_k={}/{num:03d}/'.format(fold+1, k, num=_id)
            #pathAWET =  os.path.join(awet_path,'fold_probes_{}_k={}/{num:03d}/'.format(fold+1, k, num=_id))
            #reset_directory(pathAWET, purge=True)

            #filename = '{num:03d}_{emo:s}_1.ppm'.format(num=_id, emo=emotion)
            #imsave(os.path.join(path, filename), image)
            #imsave(os.path.join(pathAWET, filename), image)

    return deidentified_images

def run_protocol_naive(deid_method, src_path, dst_path):

    filelist = read_files(path=src_path)
    images = loadImages(src_path, filelist)

    print "Deidentifying %d images using: %s" %(len(images), deid_method.__name__)
    deid = deid_method(images)

    reset_directory(dst_path)

    # save resulting images
    for name, image in zip(filelist, deid):
        imsave(join(dst_path, name), image)


def run_protocol_formal(deid_method, src_path, cluster_files, clustered_images, dst_path, k):

    from B_k_same_net import remapEmotions

    for key in cluster_files.keys():
        probe_cluster = cluster_files[key]
        probe_cluster = remapEmotions(src_path, probe_cluster)
        cluster_files[key] = probe_cluster

    print "Deidentifying %d clusters with k=%d, using: %s" %(len(cluster_files), k, deid_method.__name__)
    deidentified = deid_method(cluster_files, clustered_images, k)

    reset_directory(dst_path)

    # save resulting images
    for k in deidentified.keys():
        file_list = cluster_files[k]
        
        #print "BEFORE: ", file_list

        file_list = remapEmotions(src_path, file_list)

        #print "AFTER: ", file_list

        print "Cluster label: ", k, " size: ", len(file_list)
        
        # TODO: here we need to handle various emotions per image!
        image = deidentified[k]
        if not type(deidentified[k]) is list:
            print image.shape


        for i, name in enumerate(file_list):
            #print name
            #print type(image)
            if type(deidentified[k]) is list:
                image = deidentified[k][i]
                
            imsave(join(dst_path, name), image)
    

if __name__ == "__main__":

    # TODO: other methods: k-same pixelwise, pixelizaton, blurring
    naive_methods = [pixelize, blur]
    formal_methods = [k_same_net, k_same_M, k_same_pixel]#[k_same_net, k_same_pixel] # , k_same_M, k_same_net

    src_path = "./folds/"
    src_path_aam = "./folds_aam/"
    dst_path = "./deid/"
    cluster_cache = "./cluster_cache/"
    folds = ["fold_probes_1", "fold_probes_2", "fold_probes_3", "fold_probes_4", "fold_probes_5"]

    #from scipy.misc import imread
    #import os
    #path = "./folds/fold_probes_1/"
    #name = "005_2_1.ppm"
    #blur([imread(os.path.join(path, name))])
    #pixelize([imread(os.path.join(path, name))])
    #exit()

    do_naive = False #True
    do_formal = False #True # TODO: test k_same_M

    do_emotions = True
    emo_path = "./DB/ck+/Emotion/" # read real emotions from annotatios
    emo_gtru = readEmotions(emo_path)

    if do_formal:
        # perform clustering only once
        if not isdir(cluster_cache):
            reset_directory(cluster_cache)
            for f, fold in enumerate(folds):
                filelist = read_files(join(src_path, fold))
                for k in range(2, 11):
                    cluster_files = clusterImagesEqualGroups(join(src_path, fold), filelist, k)
                    saveClusters(cluster_files, join(cluster_cache, "clusters_f=%d_k=%d.txt" % (f+1, k)))

       # exit()


    for f, fold in enumerate(folds):

        if do_naive:

            for deidentify in naive_methods:
                source = join(src_path, fold)
                destin = join(dst_path, deidentify.__name__, fold)
                run_protocol_naive(deidentify, source, destin)

        if do_formal:
            
            for deidentify in formal_methods:
                source = join(src_path, fold)

                if deidentify.__name__ == "k_same_M": # provide alternative source of AAM aligned images
                    source = join(src_path_aam, fold) # TODO: clusters based on aam images?

                for k in range(2, 11):  
                    cluster_files = loadClusters(join(cluster_cache, "clusters_f=%d_k=%d.txt" % (f+1, k)))
                    clustered_images = loadClusteredImages(source, cluster_files)

                    destin = join(dst_path, (deidentify.__name__ + "_k=%d" % (k)), fold)
                    run_protocol_formal(deidentify, source, cluster_files, clustered_images, destin, k)

    #emotions = ['happy','angry', 'contemptuous', 'disgusted', 'fearful', 'neutral', 'sad', 'surprised']

    src_path = "./emo_folds_predicted/original/"
    cluster_cache = "./cluster_cache_emo/"
    folds = ["fold_gallery_1", "fold_gallery_2", "fold_gallery_3", "fold_gallery_4", "fold_gallery_5"]
    dst_path = "./emo_folds/"

    if do_emotions:
        if not isdir(cluster_cache):
            reset_directory(cluster_cache)
            for f, fold in enumerate(folds):
                filelist = read_files(join(src_path, fold))
                for k in range(2, 11):
                    print "Emotions: Clustered folds not available for k=%d, performing image clustering..." % (k)
                    #cluster_files = clusterImagesEqualGroups(join(src_path, fold), filelist, k)
                    cluster_files = clusterImagesDummy(join(src_path, fold), filelist, k)
                    saveClusters(cluster_files, join(cluster_cache, "clusters_f=%d_k=%d.txt" % (f+1, k)))


        for f, fold in enumerate(folds):
            source = join(src_path, fold) # TODO: clusters based on aam images?
            print "Fold: %d / %d " % (f+1, len(folds))
            for k in range(2, 11):  
                cluster_files = loadClusters(join(cluster_cache, "clusters_f=%d_k=%d.txt" % (f+1, k)))
                clustered_images = loadClusteredImages(source, cluster_files)

                destin = join(dst_path, (k_same_net.__name__ + "_k=%d" % (k)), fold)
                run_protocol_formal(k_same_net, source, cluster_files, clustered_images, destin, k)
