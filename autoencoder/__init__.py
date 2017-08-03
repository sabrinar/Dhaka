''' Variational Autoencoder based tumor subpopulation detection
    author: Sabrina Rashid 
'''
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
#from scipy.stats import norm
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn import mixture
from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics, optimizers
#from keras.datasets import mnist
from mpl_toolkits.mplot3d import Axes3D
import operator


def main(input_datafile='Synthetic_data_2500.mat',latent_dim=3,
         intermediate_dim=1024,N_starts=5,batch_size=100,learning_rate=.001,
         output_datafile='output.mat'):
#dict=scipy.io.loadmat('GSE89567')
#x_t=dict['gene_expr1']
#x_train=x_t.transpose()
#x_train=x_train[:,0:1000]
#x_train=np.concatenate((x_train,x_train[2191:2195,:]),axis=0)
##size=x_train.shape
        
    dict=scipy.io.loadmat(input_datafile)#GSE89567, cml_selectedAllCells.mat
    #x_t=dict['fg']
    x_t=dict['syn_expr'];
    #bcr_labels=dict['bcr_labels']
    #bcr_labels=bcr_labels.ravel()
    #stage1_labels=dict['stage1_labels']
    #stage1_labels=stage1_labels.ravel()
    #stage2_labels=dict['stage2_labels']
    #stage2_labels=stage2_labels.ravel()
    
    #x_train=x_t.transpose()
    #x_train=x_train[:,0:3000]
    x_train=x_t
    #x_train=np.concatenate((x_train,x_train[6281:6340,:]),axis=0)# for gse89567
    #x_train=np.concatenate((x_train,x_train[10675:10687,:]),axis=0)# forcml
    #x_train=np.concatenate((x_train,x_train[81:90,:]),axis=0)# forgm
    #x_train=np.concatenate((x_train,x_train[107:113,:]),axis=0)# forhtert
    #x_train=np.concatenate((x_train,x_train[247:253,:]),axis=0)# forxeno4
    #
    ##
    ##x_train=np.concatenate((x_train3,x_train4),axis=0)
    #x_train=x_train4
    size=x_train.shape
    
    
    #batch_size = 100
    original_dim = size[1]
    #latent_dim = 3
    #intermediate_dim = 1024
    epochs = 10
    epsilon_std = 1.0
    n_clusters=6
    #N_starts=5
    silhouette_avg=[0 for i in range(N_starts)]
    all_x_test_encoded = np.asarray([[[0 for k in range(latent_dim)] for j in range(size[0])] for i in range(N_starts)])
    all_x_test_encoded = all_x_test_encoded.astype(float)
    
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    
    
    
    # Custom loss layer
    class CustomVariationalLayer(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomVariationalLayer, self).__init__(**kwargs)
    
        def vae_loss(self, x, x_decoded_mean):
            xent_loss =  original_dim * metrics.binary_crossentropy(x, x_decoded_mean) 
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            
            return K.mean(xent_loss + kl_loss)
    
        def call(self, inputs):
            x = inputs[0]
            x_decoded_mean = inputs[1]
            loss = self.vae_loss(x, x_decoded_mean)
            
            self.add_loss(loss, inputs=inputs)
            # We won't actually use the output.
            return x
        
        
        
    for i in range(0,N_starts):
    
        x = Input(batch_shape=(batch_size, original_dim))
        h = Dense(intermediate_dim, activation='relu')(x)
        z_mean = Dense(latent_dim)(h)
        z_log_var = Dense(latent_dim)(h)
    
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    
        # we instantiate these layers separately so as to reuse them later
        decoder_h = Dense(intermediate_dim, activation='relu')
        decoder_mean = Dense(original_dim, activation='sigmoid')
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)
    
        y = CustomVariationalLayer()([x, x_decoded_mean])
        vae = Model(x, y)
        rmsprop = optimizers.rmsprop(lr=learning_rate)
        vae.compile(optimizer=rmsprop, loss=None)
    
    
        vae.fit(x_train,
                shuffle=True,
                epochs=epochs,
                batch_size=batch_size)
    
    
        # build a model to project inputs on the latent space
        encoder = Model(x, z_mean)
        x_test_encoded = encoder.predict(x_train, batch_size=batch_size)
        if np.isnan(x_test_encoded).any():
            x_test_encoded=np.asarray([[0 for j in range(latent_dim)] for i in range(size[0])])
            silhouette_avg[i]=0
        else:
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(x_test_encoded)
            silhouette_avg[i] = silhouette_score(x_test_encoded, cluster_labels)
        all_x_test_encoded[i][:][:]=x_test_encoded
        fig=plt.figure(figsize=(6, 6))
        ax3D = fig.add_subplot(111, projection='3d')
        ax3D.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], x_test_encoded[:, 2])
        plt.show()
        
        
    index, value = max(enumerate(silhouette_avg), key=operator.itemgetter(1))
    hhj=all_x_test_encoded[index][:][:]
    n_components_range = range(1, 31)
    fig=plt.figure(figsize=(6, 6))
    lowest_bic = np.infty
    bic = []
    for n_components in n_components_range:
        gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='diag',n_init=10)
        gmm.fit(hhj)
        bic.append(gmm.bic(hhj))
    
    bic = np.array(bic)+3*np.log(size[0])*n_components_range*latent_dim
    ind,val=min(enumerate(bic), key=operator.itemgetter(1))
    plt.plot(n_components_range,bic)
    
    gmm = mixture.GaussianMixture(n_components=ind, covariance_type='diag')
    gmm.fit(hhj)
    labels=gmm.predict(hhj)
    fig=plt.figure()
    ax3D = fig.add_subplot(111, axisbg="1.0",projection='3d')
    color_iter = ['navy', 'turquoise', 'cornflowerblue','darkorange','mistyrose','seagreen','hotpink','purple','thistle','darkslategray']
    for i in range(0,labels.max()+1):
        ax3D.scatter(hhj[labels==i, 0], hhj[labels==i, 1], hhj[labels==i, 2],alpha=1, color=color_iter[i])
        
#    fig=plt.figure()
#    ax3D = fig.add_subplot(111, axisbg="1.0",projection='3d')
#    hhj1=hhj[0:10688,:]    
#    labels=dict['stage2_labels']
#    labels=labels.ravel()
#    for i in range(0,labels.max()):
#       ax3D.scatter(hhj1[labels==i, 0], hhj1[labels==i, 1], hhj1[labels==i, 2],alpha=.5, color=color_iter[i])   
#    plt.legend(loc=2)
    
    scipy.io.savemat(output_datafile, {'vect':hhj})
    ## visualize obtained clusters.
    #visualizer = cluster_visualizer();
    #visualizer.append_clusters(clusters, sample);
    #visualizer.show();