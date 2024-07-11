import scipy
from keras import backend as K
import numpy as np
from keras.utils import np_utils
from scipy import misc
from keras.layers import BatchNormalization, Convolution2D, Dense, LeakyReLU, \
    Input, MaxPooling2D, merge, Reshape, UpSampling2D, Dropout
from keras.layers.merge import concatenate
from keras.models import Model
import tensorflow as tf

NUM_YALE_POSES = 10

import random
# ---- Enum classes for vector descriptions

class Emotion:
    angry = [1., 0., 0., 0., 0., 0., 0., 0.]
    anger = [1., 0., 0., 0., 0., 0., 0., 0.]

    contemptuous = [0., 1., 0., 0., 0., 0., 0., 0.]
    contempt = [0., 1., 0., 0., 0., 0., 0., 0.]

    disgusted = [0., 0., 1., 0., 0., 0., 0., 0.]
    disgust = [0., 0., 1., 0., 0., 0., 0., 0.]

    fearful = [0., 0., 0., 1., 0., 0., 0., 0.]
    fear = [0., 0., 0., 1., 0., 0., 0., 0.]

    happy = [0., 0., 0., 0., 1., 0., 0., 0.]
    neutral = [0., 0., 0., 0., 0., 1., 0., 0.]

    sad = [0., 0., 0., 0., 0., 0., 1., 0.]
    sadness = [0., 0., 0., 0., 0., 0., 1., 0.]

    surprised = [0., 0., 0., 0., 0., 0., 0., 1.]
    surprise = [0., 0., 0., 0., 0., 0., 0., 1.]

    mixed = [1.0, 0., 0., 0., 1.0, 0., 0., 1.0]

    @classmethod
    def length(cls):
        return len(Emotion.neutral)


def log10(x):
    """
    there is not direct implementation of log10 in TF.
    But we can create it with the power of calculus.
    Args:
        x (array): input array

    Returns: log10 of x

    """
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def psnr(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return -10. * log10(K.mean(K.square(y_pred - y_true)))


def build_model(identity_len=57, orientation_len=2, lighting_len=4,
                emotion_len=8, pose_len=NUM_YALE_POSES,
                initial_shape=(5, 4), deconv_layers=5, num_kernels=None,
                optimizer='adam', use_yale=False):
    """
    Builds a deconvolution FaceGen model.

    Args (optional):
        identity_len (int): Length of the identity input vector.
        orientation_len (int): Length of the orientation input vector.
        emotion_len (int): Length of the emotion input vector.
        initial_shape (tuple<int>): The starting shape of the deconv. network.
        deconv_layers (int): How many deconv. layers to use. More layers
            gives better resolution, although requires more GPU memory.
        num_kernels (list<int>): Number of convolution kernels for each layer.
        optimizer (str): The optimizer to use. Will only use default values.
    Returns:
        keras.Model, the constructed model.
    """

    if num_kernels is None:
        num_kernels = [128, 128, 96, 96, 32, 32, 16]

    # TODO: Parameter validation

    identity_input = Input(shape=(identity_len,), name='identity')

    if use_yale:
        lighting_input = Input(shape=(lighting_len,), name='lighting')
        pose_input = Input(shape=(pose_len,), name='pose')
    else:
        orientation_input = Input(shape=(orientation_len,), name='orientation')
        emotion_input = Input(shape=(emotion_len,), name='emotion')

    # Hidden representation for input parameters

    fc1 = LeakyReLU()(Dense(512)(identity_input))
    fc2 = LeakyReLU()(Dense(512)(lighting_input if use_yale else orientation_input))
    fc3 = LeakyReLU()(Dense(512)(pose_input if use_yale else emotion_input))

    params = concatenate([fc1, fc2, fc3]) #merge([fc1, fc2, fc3], mode='concat')
    #params = Dropout(rate=0.3)(params)

    params = LeakyReLU()(Dense(1024, kernel_initializer='random_normal', bias_initializer='random_normal')(params))
    #params = Dropout(rate=0.3)(params)
        
    # Apply deconvolution layers

    height, width = initial_shape

    print('height:', height, 'width:', width)

    x = LeakyReLU()(Dense(height * width * num_kernels[0], kernel_initializer='random_normal', bias_initializer='random_normal')(params))
    if K.image_data_format() == 'channels_first':
        x = Reshape((num_kernels[0], height, width))(x)
    else:
        x = Reshape((height, width, num_kernels[0]))(x)

    for i in range(0, deconv_layers):
        # Upsample input
        x = UpSampling2D((2, 2))(x)
        #x = Dropout(rate=0.3)(x)
        #x = BatchNormalization()(x)
        # Apply 5x5 and 3x3 convolutions

        # If we didn't specify the number of kernels to use for this many
        # layers, just repeat the last one in the list.
        idx = i if i < len(num_kernels) else -1
        x = LeakyReLU()(Convolution2D(num_kernels[idx], (5, 5), padding='same', kernel_initializer='random_normal',
                bias_initializer='random_normal')(x))
        x = Dropout(rate=0.3)(x)
        #x = BatchNormalization()(x)
        
        x = LeakyReLU()(Convolution2D(num_kernels[idx], (3, 3), padding='same', kernel_initializer='random_normal',
                bias_initializer='random_normal')(x))
        x = Dropout(rate=0.3)(x)
        
        x = BatchNormalization()(x)

    # Last deconvolution layer: Create 3-channel image.
    x = MaxPooling2D((1, 1))(x)
    x = UpSampling2D((2, 2))(x)
    x = LeakyReLU()(Convolution2D(8, (5, 5), padding='same', kernel_initializer='random_normal',
                bias_initializer='random_normal')(x))
    x = LeakyReLU()(Convolution2D(8, (3, 3), padding='same', kernel_initializer='random_normal',
                bias_initializer='random_normal')(x))
    x = Convolution2D(1 if use_yale else 3, (3, 3), padding='same', activation='sigmoid', kernel_initializer='random_normal',
                bias_initializer='random_normal')(x)

    # Compile the model

    if use_yale:
        model = Model(inputs=[identity_input, pose_input, lighting_input], outputs=x)
    else:
        model = Model(inputs=[identity_input, orientation_input, emotion_input], outputs=x)

    # TODO: Optimizer options
    model.compile(optimizer=optimizer, loss='msle', metrics=[psnr])

    return model


class Generator:

    def __init__(self, model_path, id_len=57, deconv_layer=6):
        model = build_model(
            identity_len=id_len,
            deconv_layers=deconv_layer,
            optimizer='adam',
            initial_shape=(5, 4),
        )
        if model_path:
            model.load_weights(model_path)
        self.id_len = id_len
        self.model = model

        print("k-Same-Net params: ", self.model.count_params())

    def getKerasModel(self):
        return self.model

    def generate(self, id, emo='neutral', orient='front', _debug=False):

        #print id

        if orient == 'front':
            orientation = np.zeros((1, 2))
        else:
            raise NotImplementedError
            
        if type(id) is list:
            id_weights = np_utils.to_categorical([id[0]], self.id_len)
            for i in id:
                id_weights[:, i] = 1.0
                id_weights = id_weights / (1.0 * len(id));
            #id_weights = np_utils.to_categorical(id, self.id_len)
        else:
            id_weights = np_utils.to_categorical([id], self.id_len)
            
        #id_weights[:, 10] = 1.0 # testing generation from multiple IDs
            
        input_vec = {
            'identity': id_weights,  # np_utils.to_categorical([id], self.id_len),
            'emotion': np.array(getattr(Emotion, emo)).reshape((1, Emotion.length())),
            'orientation': np.array(orientation),
        }

        gen = self.model.predict_on_batch(input_vec)[0]
        if K.image_data_format() == 'th':
            image = np.empty(gen.shape[2:] + (3,))
            for x in range(0, 3):
                image[:, :, x] = gen[x, :, :]
        else:
            image = gen
        image = np.array(255 * np.clip(image, 0, 1), dtype=np.uint8)
        return image

    def generate_random(self, n, multi_id=None, save_to=None):
        
        emotions = ['happy','angry', 'contemptuous', 'disgusted', 'fearful', 'neutral', 'sad', 'surprised']
    
        samples = []
        for i in range(0, n):
            emo = random.choice(emotions)
            ide = random.randint(0, 56)
            if multi_id is not None:
                ide = [random.randint(0,56) for j in range(1, multi_id)]

            image = self.generate(ide, emo)
            if save_to:
                scipy.misc.imsave(save_to + 'gen_out_' + emo + '_' + str(ide) + '.jpg', image)

            samples.append(image)

        samples = np.array(samples)
        return samples

    def generate_random_inputs(self, n, orient='front'):
        emotions = ['happy','angry', 'contemptuous', 'disgusted', 'fearful', 'neutral', 'sad', 'surprised']
    
        ids = np.empty(shape=[n, 57])
        emos = np.empty(shape=[n, 8])
        orients = np.empty(shape=[n, 2])

        inputs = []

        #print(ids.shape)

        for i in range(0, n):
            if orient == 'front':
                orientation = np.zeros((1, 2))
            else:
                raise NotImplementedError
            
            id = random.randint(0, 56)
            emo = random.choice(emotions)
            # TODO: can be multiple ids?

            if type(id) is list:
                id_weights = np_utils.to_categorical([id[1]], self.id_len)
                for i in id:
                    id_weights[:, i] = 1.0
                    id_weights = id_weights / (1.0 * len(id));
                #id_weights = np_utils.to_categorical(id, self.id_len)
            else:
                id_weights = np_utils.to_categorical([id], self.id_len)
                
            #id_weights[:, 10] = 1.0 # testing generation from multiple IDs
                
            input_vec = {
                'identity': id_weights,  # np_utils.to_categorical([id], self.id_len),
                'emotion': np.array(getattr(Emotion, emo)).reshape((1, Emotion.length())),
                'orientation': np.array(orientation),
            }

            #print(input_vec['identity'][:].shape)

            ids[i, :] = input_vec['identity']
            emos[i, :] = input_vec['emotion']
            orients[i, :] = input_vec['orientation']

            #inputs.append(np.array(id_weights, np.array(getattr(Emotion, emo)).reshape((1, Emotion.length()), np.array(orientation)))
            inputs.append([input_vec['identity'], input_vec['emotion'], input_vec['orientation']])

        #input_vec = [ids, emos, orients]
        input_vec = { #[np.array(ids), np.array(emos), np.array(orients)]
            'identity': ids,  # np_utils.to_categorical([id], self.id_len),
            'emotion': emos, #np.array(getattr(Emotion, emo)).reshape((1, Emotion.length())),
            'orientation': orients,
        }
        return input_vec
        #return inputs

    def generate_actual(self, save_to=None): # TODO: generalize for each k?
        emotions = ['happy','angry', 'contemptuous', 'disgusted', 'fearful', 'neutral', 'sad', 'surprised']
    
        samples = []
        for emo in emotions:    
            for i in range(0, 57): # 3x k=2
                # k=2
                #ids = [i, i-1]; # , i-1
                image = self.generate(i, emo)
                if save_to:
                    scipy.misc.imsave(save_to + 'gen_out_' + emo + '_' + str(i) + '.jpg', image)

                samples.append(image)

        samples = np.array(samples)
        return samples

    def __str__(self):
        # TODO: print in shape, out shape
        return "{}".format(self.__class__.__name__)


if __name__ == '__main__':
#    gen = Generator('./output/FaceGen.RaFD.model.d6.adam.iter500.h5')
   # gen = Generator('./output/FaceGen.RaFD.model.d3.adam.h5', deconv_layer=3)
    gen = Generator('./models/FaceGen.RaFD.model.d5.adam.h5', deconv_layer=5)
    
    emotions = ['happy','angry', 'contemptuous', 'disgusted', 'fearful', 'neutral', 'sad', 'surprised']
    
    for emotion in emotions:
    
        
        for i in range(0, 3): # 3x k=2
            # k=2
            #ids = [i, i+2];
            ids = [10 + i*10, 11 + i*10];
            image = gen.generate(ids, emotion)
            scipy.misc.imsave('../out/gen_out_' + emotion + '_' + '-'.join([str(id) for id in ids]) + '.jpg', image)
            
            
    
        for i in range(0, 2): # 2x k=3
            # k=1
            #image = gen.generate(i, emotion)
            #scipy.misc.imsave('../out/gen_out_' + emotion + '_' + str(i) + '.jpg', image)
            
            # k=4
            #ids = [i, i+2, i+4, i+6];
            #ids = [52, 53, 54];
            ids = [10 + i*10, 11 + i*10, 12 + i*10];
            image = gen.generate(ids, emotion)
            scipy.misc.imsave('../out/gen_out_' + emotion + '_' + '-'.join([str(id) for id in ids]) + '.jpg', image)
