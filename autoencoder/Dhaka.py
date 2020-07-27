# Copyright (c) Microsoft Corporation
# All rights reserved. 
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and 
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING 
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

''' Variational Autoencoder based tumor subpopulation detection
    author: Sabrina Rashid 
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import mixture
from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics, optimizers
from mpl_toolkits.mplot3d import Axes3D
import operator



def my_except_hook(exctype, value, traceback):
        print('There has been an error in the system')
sys.excepthook = my_except_hook

def Dhaka(input_datafile='oligo_malignant.txt',latent_dim=3,
         N_starts=5,batch_size=100,learning_rate=.0001, epochs = 5,
         clip_norm=2,output_datafile='output',to_cluster= 0,n_genes=5000,
         gene_selection=0,selection_criteria='average',to_plot=1,verbose=True,
         relative_expression=0, activation='sigmoid'):
    
    
    # read datafile
    x_t=np.loadtxt(input_datafile)    
#    dict=scipy.io.loadmat(input_datafile)
#    x_t=dict['syn_expr'] # input expression matrix # of cells * # of genes
    x_t[np.isnan(x_t)]=0
    orig_size=x_t.shape
    
    if orig_size[1]<1000:
        print('Number of genes too small')

    
    ## check input parameters
    
    # Latent dim
    if not (type(latent_dim) is int):
        raise TypeError('Latent dimensions must be an integer')
    if latent_dim<1:
        raise ValueError('Latent dimensions should be atleast 1')
    elif latent_dim>256:
        raise ValueError('Latent dimensions should be less than 256')
        
    # N_starts
    if not (type(N_starts) is int):
        raise TypeError('Number of warm starts must be an integer')
    elif N_starts<1:
        raise ValueError('Number of warm starts must be more than 1')
    elif N_starts>50:
        raise Warning('Number of warm starts more than 50. Should take a long time to run.')    
        
    # batch_size
    if not (type(batch_size) is int):
        raise TypeError('Batch size must be an integer')
    elif batch_size==0:
        raise ValueError('Batch size should not be zero')
    elif batch_size>orig_size[0]:
        raise ValueError('Batch size should not be larger than the total number of cells')    
    
    # n_genes
    if not (type(n_genes) is int):
        raise TypeError('Number of genes must be an integer')
    elif n_genes<1000:
        print('Number of genes too small, Encoding might not be optimal')
    elif n_genes>orig_size[1]:
        n_genes=orig_size[1]
        print('Using all the genes in the dataset')
        
    #epochs
    if not (type(epochs) is int):
        raise TypeError('Number of epochs must be an integer')
    elif epochs<1:
        raise ValueError('Number of epochs should be atleast 0')
    elif epochs>100:
        print('Very large number of epochs, training should take a lot of time')
    
    # output_datafile
    if not (type(output_datafile) is str):
        raise TypeError('Output datafile name should be a string')
    
    if learning_rate>.1:
        print('Learning rate too high')
    
    if clip_norm>10:
        print('Clip norm too high')

    
    # gene selection
    if gene_selection:
        a=np.zeros((orig_size[1]))  #[0 for i in range(size[1])]
        cv=np.zeros((orig_size[1]))
        en=np.zeros((orig_size[1]))
        for i in range(0,orig_size[1]):
            cv[i]=np.std(x_t[:,i])/np.mean(x_t[:,i]) # CV criteria
            a[i]=sum(x_t[:,i]) # average value
            hist, bin_edges=np.histogram(x_t[:,i],bins=100)
            pk=hist/sum(hist)
            en[i]=scipy.stats.entropy(pk)    # entropy        
        if selection_criteria=='average':
            sorted_indices=sorted(range(len(a)), key=lambda k: a[k],reverse=True)
        elif selection_criteria == 'cv':
            sorted_indices=sorted(range(len(cv)), key=lambda k: cv[k],reverse=True)
        elif selection_criteria == 'entropy':
            sorted_indices=sorted(range(len(en)), key=lambda k: en[k],reverse=True)            
        else:
            raise ValueError('Not a valid selection criteria, Refer to the readme file for valid selection criteria')
            
        x_t=x_t[:,sorted_indices[0:min(n_genes,orig_size[1])]]
    
    if relative_expression:
        y=np.mean(x_t,axis=1)
        print(y.shape)
        if gene_selection:
            x_t=x_t-np.tile(y,(n_genes,1)).transpose()
        else:
            x_t=x_t-np.tile(y,(orig_size[1],1)).transpose()

        
    x_train=x_t   

    # pad end cells for being compatible with batch size
    reminder=orig_size[0]%batch_size
    x_train=np.concatenate((x_train,x_train[(orig_size[0]-batch_size+reminder):orig_size[0],:]),axis=0)
    size=x_train.shape
    
    
    # internal parameters
    original_dim = size[1]
    epsilon_std = 1.0
    n_clusters=6
    intermediate_deep_dim=1024
    intermediate_deep_dim2=512
    intermediate_dim = 256
    color_iter = ['navy', 'turquoise', 'cornflowerblue','darkorange','mistyrose','seagreen','hotpink','purple','thistle','darkslategray']
    
    # required initializations
    silhouette_avg=np.zeros((N_starts))#[0 for i in range(N_starts)]
    all_x_encoded = np.zeros((N_starts,size[0],latent_dim))#np.asarray([[[0 for k in range(latent_dim)] for j in range(size[0])] for i in range(N_starts)])
    all_x_encoded = all_x_encoded.astype(float)
    
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
            return x
        
        
        
    for i in range(0,N_starts):
    
        x = Input(batch_shape=(batch_size, original_dim))
        e = Dense(intermediate_deep_dim, activation = 'relu')(x)
        d= Dense(intermediate_deep_dim2, activation ='relu')(e)
        h = Dense(intermediate_dim, activation='relu')(d)
    
        z_mean = Dense(latent_dim)(h)
        z_log_var = Dense(latent_dim)(h)

        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

        # we instantiate these layers separately so as to reuse them later
        decoder_h = Dense(intermediate_dim, activation='relu')
        decoder_d = Dense(intermediate_deep_dim2, activation ='relu')
        decoder_e = Dense(intermediate_deep_dim, activation = 'relu')
        decoder_mean = Dense(original_dim, activation=activation)
        h_decoded = decoder_h(z)
        d_decoded = decoder_d(h_decoded)
        e_decoded = decoder_e(d_decoded)
        x_decoded_mean = decoder_mean(e_decoded)

        y = CustomVariationalLayer()([x, x_decoded_mean])
        vae = Model(x, y)
        rmsprop = optimizers.rmsprop(lr=learning_rate,clipnorm=clip_norm)
        vae.compile(optimizer=rmsprop, loss=None)
    
    
        vae.fit(x_train,
                shuffle=True,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose)
    
    
        # build a model to project inputs on the latent space
        encoder = Model(x, z_mean)
        x_encoded = encoder.predict(x_train, batch_size=batch_size)
        if np.isnan(x_encoded).any():
           # x_encoded=np.asarray([[0 for j in range(latent_dim)] for i in range(size[0])])
            silhouette_avg[i]=0
        else:
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(x_encoded)
            silhouette_avg[i] = silhouette_score(x_encoded, cluster_labels)

        all_x_encoded[i][:][:]=x_encoded
        

    index, value = max(enumerate(silhouette_avg), key=operator.itemgetter(1))
    x_encoded_final=all_x_encoded[index][:][:]
    x_encoded_final=x_encoded_final[0:orig_size[0],:]
    
    if np.isnan(x_encoded_final).any():
        print(np.isnan(x_encoded_final).any())
        raise Warning('NaNs, check input, learning rate, clip_norm parameters')
    
    if to_plot:
        if latent_dim>=3:
            fig=plt.figure(figsize=(6, 6))
            ax3D = fig.add_subplot(111, projection='3d')
            ax3D.scatter(x_encoded_final[:, 0], x_encoded_final[:, 1], x_encoded_final[:, 2])
            ax3D.set_xlabel('Latent dim 1')
            ax3D.set_ylabel('Latent dim 2')   
            ax3D.set_zlabel('Latent dim 3')            
            plt.savefig(output_datafile+'fig_projection.png')
        elif latent_dim==2:
            fig=plt.figure(figsize=(6, 6))
            plt.scatter(x_encoded_final[:, 0], x_encoded_final[:, 1])
            plt.xlabel('Latent dim 1')
            plt.ylabel('Latent dim 2') 
            plt.savefig(output_datafile+'fig_projection.png')
        elif latent_dim==1:
            n_range = range(0, orig_size[0])
            fig=plt.figure(figsize=(6, 6))
            plt.plot(n_range,x_encoded_final)
            plt.xlabel('Cells')
            plt.ylabel('Latent dim 1')
            plt.savefig(output_datafile+'fig_projection.png') 
        
    if to_cluster:
        n_components_range = range(1, 10)
        bic = []
        for n_components in n_components_range:
            gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='tied',n_init=10)
            gmm.fit(x_encoded_final)
            bic.append(gmm.bic(x_encoded_final))
    
        bic = np.array(bic)+np.log(size[0])*n_components_range*latent_dim
        ind,val=min(enumerate(bic), key=operator.itemgetter(1))
        if to_plot:
            fig=plt.figure(figsize=(6, 6))
            plt.plot(n_components_range,bic)
            plt.xlabel('Number of clusters')
            plt.ylabel('BIC')
            plt.savefig(output_datafile+'fig_bic.png')
    
        gmm = mixture.GaussianMixture(n_components=ind+1, covariance_type='tied')
        gmm.fit(x_encoded_final)
        labels=gmm.predict(x_encoded_final)
        
        if to_plot:
            if latent_dim>=3:
                fig=plt.figure()
                ax3D = fig.add_subplot(111, axisbg="1.0",projection='3d')        
                for i in range(0,labels.max()+1):
                    ax3D.scatter(x_encoded_final[labels==i, 0], x_encoded_final[labels==i, 1], x_encoded_final[labels==i, 2],alpha=1, color=color_iter[i])
                    ax3D.set_xlabel('Latent dim 1')
                    ax3D.set_ylabel('Latent dim 2')   
                    ax3D.set_zlabel('Latent dim 3')
                    plt.savefig(output_datafile+'fig_cluster.png')

            elif latent_dim==2:
                fig=plt.figure()
                for i in range(0,labels.max()+1):
                    plt.scatter(x_encoded_final[labels==i, 0], x_encoded_final[labels==i, 1],alpha=1, color=color_iter[i])
                    plt.xlabel('Latent dim 1')
                    plt.ylabel('Latent dim 2') 
                    plt.savefig(output_datafile+'fig_cluster.png')
        
        #scipy.io.savemat(output_datafile+'.mat', {'vect':x_encoded_final,'labels':labels,'bic':bic})
        np.savetxt(output_datafile+'labels.txt',labels)
        np.savetxt(output_datafile+'bic.txt',bic)
    
    #else:
        #scipy.io.savemat(output_datafile+'.mat', {'vect':x_encoded_final})
    np.savetxt(output_datafile+'.txt',x_encoded_final)
