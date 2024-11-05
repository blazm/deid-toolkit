import os
import cv2
import argparse
from k_same_net.Generator import Emotion
import csv
from tqdm import tqdm

def make_gen():
    
    from keras import backend as K
    K.set_image_data_format('channels_last')
    from k_same_net.Generator import Generator

    #do_emotions = False

    deconv_layer = 6 # 5 or 6
    model_name = 'FaceGen.RaFD.model.d{}.adam'.format(deconv_layer)
#    model_path = '../de-id/generator/output/{}.h5'.format(model_name)
    model_path = os.path.join('k_same_net', 'models', '{}.h5'.format(model_name)) # locally stored models

    gen = Generator(model_path, deconv_layer=deconv_layer)
    return gen

def k_same_net(gen, clustered_probes=None, clustered_images=None, k=2, emotion = 'neutral'):

    #proxy_path = "../de-id/DB/rafd2-frontal/"
    #proxys = read_files(proxy_path)
    #proxys = filterProxy(proxys)

    import random
    ids = [random.randint(0, 56)]*k

    image = gen.generate(ids, emotion)

    #del gen

    return image

def main(dataset_path,dataset_save, dataset_filetype = 'jpg',dataset_newtype = 'jpg'):
    img_names = [i for i in os.listdir(dataset_path) if dataset_filetype in i] # change ppm into jpg
    img_names.sort()
    img_paths = [os.path.join(dataset_path, i) for i in img_names]
    dataset_name = os.path.basename(dataset_save)
    labels_path = os.path.join("..","datasets","labels",dataset_name+"_labels.csv")
    emotion_code = 0
    emotion_dict = {'0': "neutral", "1":"anger", "3": "contempt", "4": "disgust", "5":"fear","6":"happy","7":"sad","8":"surprise"}
    #emotion_dict = {'Neutral':0,'Anger':1,'Scream':2,'Contempt':3,'Disgust':4,'Fear':5,'Happy':6,'Sadness':7,'Surprise':8}
    def ensure_dir(d):
        #dd = os.path.dirname(f)
        if not os.path.exists(d):
            os.makedirs(d)
    ensure_dir(dataset_save)

    # TODO: go over all files
    gen = make_gen()
    
    for img_name, img_path in tqdm(zip(img_names, img_paths), total=len(img_names)):  
        if os.path.exists(os.path.join(dataset_save, img_name)):
            #print("File already exists, skipping: ", img_name )
            continue
        #print("Processing: ", img_name)
        emo = ''
        with open(labels_path,"r") as labels:
            reader=csv.DictReader(labels)
            for row in reader:
                if row['Name']==img_name and row['Emotion_code']!='':
                    emotion_code = row['Emotion_code']
                    try:
                        if dataset_name == 'arface' and emotion_code=='2':
                            emo = emotion_dict['5']
                        else:
                            emo = emotion_dict[emotion_code]
                    except:
                        print("ERRO####################################")
                        print(dataset_name)
                        print(emotion_code)
                        print("ERRO####################################")
                        raise Exception("Something went wrong!")
                    #print("Emotion : ", emo )
                    break
        if emo=='':
            print("No emotion label available, deidentification will run with neutral expression")
            emo ='neutral'

        deid_img = k_same_net(gen, k=5, emotion=emo)
        deid_img = cv2.cvtColor(deid_img, cv2.COLOR_BGR2RGB)

        cv2.imwrite(os.path.join(dataset_save, img_name), deid_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process and anonymize images.")
    parser.add_argument('dataset_path', type=str, help="Path to the dataset directory")
    parser.add_argument('dataset_save', type=str, help="Path to the save directory")
    parser.add_argument('--dataset_filetype', type=str, default='jpg', help="Filetype of the dataset images (default: jpg)")
    parser.add_argument('--dataset_newtype', type=str, default='jpg', help="Filetype for the anonymized images (default: jpg)")

    args = parser.parse_args()
    main(args.dataset_path, args.dataset_save, args.dataset_filetype, args.dataset_newtype)


    