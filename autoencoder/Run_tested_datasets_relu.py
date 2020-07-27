# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 16:14:05 2018

@author: Sabrina
"""

from Dhaka import Dhaka
import parula as par
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


color_map=par.parula_map


# run oligodendroglioma with selected genes
output_datafile='output_oligo_sig'
Dhaka(input_datafile='Oligo_malignant_significant_genes.txt', output_datafile=output_datafile, relative_expression=1,
      N_starts=5,batch_size=20,gene_selection=0,epochs=5,activation='relu')

# scatterplot latent representation with differentiation and lineage metric
x_encoded_final=np.loadtxt(output_datafile+'.txt')
score_file='Oligo_score.txt'   

Score=np.loadtxt(score_file)
fig=plt.figure(figsize=(6, 6))
ax3D = fig.add_subplot(111, projection='3d')
sc=ax3D.scatter(x_encoded_final[:, 0], x_encoded_final[:, 1], x_encoded_final[:, 2],s=30,alpha=1, c=Score[:,0],cmap=color_map)
ax3D.set_xlabel('Latent dim 1')
ax3D.set_ylabel('Latent dim 2')   
ax3D.set_zlabel('Latent dim 3')  
plt.title('Oligodendroglioma with signature genes, Lineage score')            
plt.colorbar(sc)

fig=plt.figure(figsize=(6, 6))
ax3D = fig.add_subplot(111, projection='3d')
sc=ax3D.scatter(x_encoded_final[:, 0], x_encoded_final[:, 1], x_encoded_final[:, 2],s=30,alpha=1, c=Score[:,1],cmap=color_map)
ax3D.set_xlabel('Latent dim 1')
ax3D.set_ylabel('Latent dim 2')   
ax3D.set_zlabel('Latent dim 3')    
plt.title('Oligodendroglioma with signature genes, Differentiation score')   
plt.colorbar(sc)

# run oligodendroglioma with auto selected genes
output_datafile='output_oligo'
Dhaka(input_datafile='Oligo_malignant.txt', output_datafile=output_datafile, relative_expression=1,
      N_starts=1,batch_size=50,gene_selection=1,epochs=20,activation='relu')

# scatterplot latent representation with lineage metric
x_encoded_final=np.loadtxt(output_datafile+'.txt')
score_file='Oligo_score.txt'   

Score=np.loadtxt(score_file)
fig=plt.figure(figsize=(6, 6))
ax3D = fig.add_subplot(111, projection='3d')
sc=ax3D.scatter(x_encoded_final[:, 0], x_encoded_final[:, 1], x_encoded_final[:, 2],s=30,alpha=1, c=Score[:,0],cmap=color_map)
ax3D.set_xlabel('Latent dim 1')
ax3D.set_ylabel('Latent dim 2')   
ax3D.set_zlabel('Latent dim 3')  
plt.title('Oligodendroglioma with auto selected genes, lineage score')          
plt.colorbar(sc) 



# run Glioblastoma 
output_datafile='output_glio'
Dhaka(input_datafile='Glioblastoma.txt', output_datafile=output_datafile, relative_expression=0,
      N_starts=5,batch_size=200,gene_selection=0,epochs=5,activation='relu')
#scatterplot with stemness score
x_encoded_final=np.loadtxt(output_datafile+'.txt')
score_file='Glioblastoma_score.txt'   

Score=np.loadtxt(score_file)
fig=plt.figure(figsize=(6, 6))
ax3D = fig.add_subplot(111, projection='3d')
sc=ax3D.scatter(x_encoded_final[:, 0], x_encoded_final[:, 1], x_encoded_final[:, 2],s=30,alpha=1, c=Score,cmap=color_map)
ax3D.set_xlabel('Latent dim 1')
ax3D.set_ylabel('Latent dim 2')   
ax3D.set_zlabel('Latent dim 3')
plt.title('Glioblastoma, Stemness score')              
plt.colorbar(sc)
# run Melanoma
output_datafile='output_mel'
Dhaka(input_datafile='Melanoma_malignant.txt', output_datafile=output_datafile, relative_expression=1,
      N_starts=1,batch_size=20,gene_selection=1,n_genes=5000,epochs=4,activation='relu')
# scatterplot with MITF-AXL score
x_encoded_final=np.loadtxt(output_datafile+'.txt')
score_file='Melanoma_score.txt'   

Score=np.loadtxt(score_file)
fig=plt.figure(figsize=(6, 6))
ax3D = fig.add_subplot(111, projection='3d')
sc=ax3D.scatter(x_encoded_final[:, 0], x_encoded_final[:, 1], x_encoded_final[:, 2],s=30,alpha=1, c=Score,cmap=color_map)
ax3D.set_xlabel('Latent dim 1')
ax3D.set_ylabel('Latent dim 2')   
ax3D.set_zlabel('Latent dim 3')   
plt.title('Melanoma, MITF-AXL score')   
plt.colorbar(sc)

# run copy number data
output_datafile='output_copy'
Dhaka(input_datafile='Single_cell_copy.txt', output_datafile=output_datafile, relative_expression=0,
      N_starts=5,batch_size=10,gene_selection=1,n_genes=1000,epochs=5,activation='relu')

# scatterplot with Xenograft time label
x_encoded_final=np.loadtxt(output_datafile+'.txt')
fig=plt.figure(figsize=(6, 6))
ax3D = fig.add_subplot(111, projection='3d')
ax3D.scatter(x_encoded_final[:260, 0], x_encoded_final[:260, 1], x_encoded_final[:260, 2],alpha=.5, color='blue')
ax3D.scatter(x_encoded_final[260:, 0], x_encoded_final[260:, 1], x_encoded_final[260:, 2],alpha=.5, color='red')
ax3D.set_xlabel('Latent dim 1')
ax3D.set_ylabel('Latent dim 2')   
ax3D.set_zlabel('Latent dim 3')
plt.title('Single cell copy number data')
plt.legend(['Xenograft 3','Xenograft 4'])  



