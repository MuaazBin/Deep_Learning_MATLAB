%% Author: Muaaz Bin Sarfaraz , Chadi El Hajj

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

%% Multi SVM 
%% Grid Search with different Kernel function , Box constraint and kernel scale
%% Takes approximately 30 minutes for grid search Uncomment if required from
%% line 31 till line 64

% KF={'linear','rbf'};% Trying two different Kernel Functions
% BC=[0.05 0.075 0.1 0.5];%Box Constraint values for Grid Search
% KS=[1 7.5 10 15];% Values for Kernel Scale
% error_linear=zeros(4,4); % Empty arrays for storing Grid Search results
% error_rbf=zeros(4,4);
% time_linear=zeros(4,4);
% time_rbf=zeros(4,4);
% 
% for i=1:2  % loop for iteraing kernel functions
%     for a=1:4 % loop for iteratingbox constraint
%          for b=1:4 %loop for iterating Kernel Scale
%             tic; %starting stop watch for recording time
%             t = templateSVM('SaveSupportVectors',true,'Standardize',1,'KernelFunction',KF{i},...
%             'BoxConstraint',BC(a),'KernelScale',KS(b)); %Initializing SVM
%           
% 
%             SVM = fitcecoc(TrainX,TrainY,'Learners',t); % Training Model
% 
%             CVMdl = crossval(SVM,'kfold',3); % 3-fold CV
% %           validationError = (kfoldLoss(CVMdl, 'LossFun', 'ClassifError'))*100;
%           if i==1 % Storing values in Linear kernel variable
%               error_linear(a,b)= kfoldLoss(CVMdl)
%               time_linear(a,b)=toc
%           else  % Storing values in RBF kernel variable
%               error_rbf(a,b)= kfoldLoss(CVMdl)
%               time_rbf(a,b)=toc %stopwatch stopped
%           end;
%          end;
%     end;      
% end;
% save error_linear; % saving grid search results
% save error_rbf;
% save time_linear;
% save time_rbf;

%% Final Model SVM
% Parameters: BC 0.5 , KS 10 , Kernel Linear

 t = templateSVM('SaveSupportVectors',true,'Standardize',1,'KernelFunction','linear',...
          'BoxConstraint',0.5,'KernelScale',10); %setting optimal parameters in final model

SVM = fitcecoc(TrainX,TrainY,'Learners',t); % training again on Training Set
CVMdl = crossval(SVM,'kfold',3);% 3 fold corss validation
validationError = (kfoldLoss(CVMdl, 'LossFun', 'ClassifError'))*100;%cross validation error
SVM_Response = predict(SVM,TestX);% predicting Test Set labels
C_mat_SVM=confusionmat(TestY,SVM_Response,'order',unique(SVM_Response));% plotting confusion Matrix
heatmap(C_mat_SVM, unique(SVM_Response), unique(SVM_Response), 1,'Colormap','red','ShowAllTicks',1,'UseLogColorMap',true,'Colorbar',true);
%Plotting a confusion matrix using customized MATLAB code