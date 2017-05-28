%% Author Chadi El Hajj and Muaaz Bin Sarfaraz
%% Using Sparse Autoencoder to attempt boosting SVM classifcation accuracy on MNIST

close all
clear all
clc
load MNISTTestX.mat; % loads complete MNIST Test Set 10,000 x  784 dimensions
load MNISTTestY.mat; %loads complete MNIST Test Set 10,000 x  1 labels
load mnistTX.mat;% loads complete MNIST Training Set 60,000 x  784 dimensions
load mnistTY.mat; % loads complete MNIST Training Set 60,000 x  1 labels

rng('default') % random number generator default for replicating same results as shown in report
rndTrainIx= randperm(60000,6000);% to generate random subset of 6000 samples from MNIST training dataset
rndTestIx= randperm(10000,1000);% to generate random subset of 1000 samples from MNIST test dataset

%% Training Set of 6,000 samples extracted randomly from 60,000 datasamples in MNIST
TrainX= Xtrain(rndTrainIx(1:6000), :); %features
TrainY=Ytrain(rndTrainIx(1:6000), :);%labels

[a,b]=hist(TrainY,unique(TrainY)); % Checking all labels(0...9) present in almost equal ratio

%% Test set of 1,000 samples of unseen digits extracted from MNIST's 10,000 test samples
TestX= Xtest(rndTestIx(1:1000), :); %features
TestY= Ytest(rndTestIx(1:1000), :); %labels

clear Ytrain % removing extra variables from workspace
clear Ytest
clear Xtrain
clear Xtest

%% Sparse Autoencoder
%% Preprocessing the input vector with Sparse AutoENcoder before passing to SVM or MLP Model for classification
%Encoder-1 784 to 100
% Training it would take 15 minutes
autoenc1 = trainAutoencoder(TrainX',100,'MaxEpochs',10 ,'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false); % Training an Autoencoder 784 to 100 Neurons in Hidden Layer
% 784 input and output nodes

xReconstructed = predict(autoenc1,TestX'); %Reconstruct Test data
TrainX_Reconstructed=predict(autoenc1,TrainX'); %Reconstruct Training data

figure                                          % plot actual images
colormap(gray)                                  % set to grayscale

for i = 1:25                                    % preview first 25 samples
    subplot(5,5,i)                              % plot them in 6 x 6 grid
    digit = reshape(TrainX(i, 1:end), [28,28])';    % row = 28 x 28 image
    imagesc(digit)                              % show the image
    title(num2str(TrainY(i, 1)))                    % show the label
end

figure                                          % plot regenerative images
colormap(gray)                                  % set to grayscale
TrainX_Reconstructed=TrainX_Reconstructed';
for i = 1:25                                    % preview first 25 samples
    subplot(5,5,i)                              % plot them in 6 x 6 grid
    digit = reshape(TrainX_Reconstructed(i, 1:end), [28,28])';    % row = 28 x 28 image
    imagesc(digit)                              % show the image
    title(num2str(TrainY(i, 1)))                    % show the label
end

%%  Final   AutoEncoder with SVM for classification
% Parameters: BC 0.5 , KS 10 , Kernel Linear
 
t = templateSVM('SaveSupportVectors',true,'Standardize',1,'KernelFunction','linear',...
          'BoxConstraint',0.5,'KernelScale',10); %setting optimal parameters in final model

SVM = fitcecoc(TrainX_Reconstructed,TrainY,'Learners',t); % training again on Training Set
SVM_Responsetest = predict(SVM,xReconstructed');% predicting Test Set labels
SVM_Responsetrain = predict(SVM,TrainX_Reconstructed);% predicting Training Set labels

error_train=(1- sum((TrainY) == SVM_Responsetrain) / length(SVM_Responsetrain))*100 % Training Error
error_test=(1- sum((TestY) == SVM_Responsetest) / length(SVM_Responsetest))*100% Test Error

%Error increased using Autoencoder