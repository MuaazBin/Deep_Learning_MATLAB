Author: Muaaz Bin Sarfaraz
	Chadi El Hajj

Please copy all files to Matlab Directory

**Main Files:

1. MLP_Model_1 

Initial Data Analysis
Contains the main Multi Layer Perceptron Model & Grid Search trained with pattern net and 3 fold cross val with cross entropy
Error on Test Set : 9.5 %

##Grid Search is commented would take 30 minutes to run code if uncommented


2. Multi_SVM_Model_2

Contains grid search for Multiclass SVM,and Final SVM Model with linear kernel 
Error on Test Set : 8.233 %

##Grid Search is commented would take 30 minutes to run code if uncommented

3. Autoencoder_SVM

As recommended by Arthur, extension of SVM with unsupervised Autoencoder based preprocessing on data
and using Final SVM model from Part 2 for testing.
No Grid search or fine tuning performed

Error on Test Set : 21 %

***Other Files:

MNISTTestX is the MNIST Test Set Features
MNISTTestY is the MNIST Test Set Labels
mnistTX is MNIST training Features
MNISTTY is the MNIST Training Labels

my_net is an array of nets saved
my_net{2,3} is the best final net

heatmap is a file used for creating confusion matrix heatmao (downlaoded from Mathworks)
drtoolbox is a dimension reduction tool box by laurens used for initial data analysis PCA and t-SNE