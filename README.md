# Project Dhaka
Python module applying deep learning to improve clustering and other analysis of single-cell genomic data (gene expression and copy number variation).

Inputs:

input_datafile: Should be a txt file containing the expression/copy number matrix. Rows=cells, Cols=Genes
                Should have atleast 1000 genes
                Considers input as log2 transformed 
                NaNs will be replaced with zeros
Sample data_file: https://drive.google.com/file/d/1dIBkt1RGiv4Vh6GgFw04beHm6BgSoOhn/view?usp=sharing

latent_dim: should be integer input between 2 to 256, default = 3


N_starts: should be integer input between 1 to 50, default =1

batch_size: should be integer input between 10 to total number of cells, default:100


learning_rate: should be between 0.01 to 0.00001. default: 0.0001


clip_norm: should be between .5 to 3. default: 2


epochs: should be integer input between 1 to 100, default: 5


output_datafile: name of the outputfile without extension, save name will be used as prefix if plots are to be saved


to_cluster: should be 0 or 1, default: 1


gene_selection: should be 0 or 1, default: 1


n_genes: Number of geens to be selected when gene_selection==1


selection_criteria: criteria to select genes, possible options 'cv', 'entropy', 'average', default:average
                     for mathematical formulation of the formula refer to the publication
                     
                     
to_plot: should be 0 or 1. for 1 plots will be saved as .png with output_datafile name prefix
         in case of latent dimensions more than 3, the first three dimensions (unlike PCA the dimensions are not ranked) will be plotted            in the scattered plot,but all the dimensions will be stored in the datafile for further manipulation.
         
         
relative_expression: should be 0 or 1. refer publication for formulation


activation: should be either 'sigmoid' or 'relu'. Default: 'sigmoid'

Outputs

encoded features will be saved in output_datafile.txt : Rows=cells, Cols=Latent dims


if to_plot ==1, corresponding projection figure will be saved as png


if to_cluster==1, cluster labels will be saved as output_datafilelabels.txt
                  bic value for k=1:10 will be saved in output_datafilebic.txt
                  scattered plot colored by predicted labels will be saved as .png
                  

         

