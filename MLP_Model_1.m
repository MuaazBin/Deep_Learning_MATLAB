%% Author Chadi El Hajj and Muaaz Bin Sarfaraz
% Dimension Reduction Tool downloaded from https://lvdmaaten.github.io/drtoolbox/
% for performing t-SNE and PCA in initial data analysis
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
%% Initial Data Visualisation & Analysis

% Since MNIST is a high dimensional data set, its dimension is reduced
%and then displayed using scatter plot for initial data analysis

% % Visualisation 1: PCA
figure;
pc = pca(TrainX');
gscatter(pc(:,1), pc(:,2),TrainY)
xlabel('PC 1');
ylabel('PC 2');
title('PCA of handwritten digits(MNIST)');
grid on;
 
% % Visualisation 2: T-SNE
% % (Time taken 15 %minutes on Acer Aspire S3 corei7 4GB ram)

% % Uncomment next line for t-SNE visualisation takes about 15 minutes
%mappedX = tsne(TestX, [], 2, 50, 20);% reduced to two dimesnions

%t-SNE is an un-supervised dimension reduction technique by Geoffrey Hinton 
% and Laurens van der Maaten, which is successful in clustering all digits
% in MNIST presented in Google talks.
% Laurens van der Maaten matlab drtoolbox used

% % Uncomment next line for t-SNE visualisation takes about 15 minutes
%gscatter(mappedX(:,1), mappedX(:,2), TestY)% colored by labels, for visualisation

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Data Preprocessing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TestY(TestY==0)=10; % Preprocessing: replacing 0 with 10 for handling labels, since indexing starts with 1 in matlab NPRTOOLmy
TrainY(TrainY==0)=10;  % Preprocessing: replacing 0 with 10 for handling labels since indexing starts with 1 in matlab NPRTOOL
Test_dY=dummyvar(TestY);%creating an indicator matrix for Neural Net Pattern recognition target selection
Train_dY=dummyvar(TrainY);%creating an indicator matrix for Neural Net Pattern recognition target selection

TestX= TestX'; %transposing for NN App
TestY=TestY';
TrainX=TrainX';
TrainY=TrainY';
Train_dY=Train_dY';
Test_dY=Test_dY';


%% FeedForward net with one hidden layers 
%% Grid Search for optimal parameters
% Takes 30 minutes Uncomment for running
% error_val=zeros(3,1);
% time=zeros(3,1);
% error_train=zeros(3,1);
% MSE=zeros(3,1);
% 
% neurons=[25 50 100 200];
% epochs=[100 300 700];
% lr=[0.01 0.9];
% momentum=[0.1 0.9];
% 
% error11=zeros(4,3);
% error12=zeros(4,3);
% error21=zeros(4,3);
% error22=zeros(4,3);
% 
% my_net=cell(size(error21));
% 
% errortrain11=zeros(4,3);
% errortrain12=zeros(4,3);
% errortrain21=zeros(4,3);
% errortrain22=zeros(4,3);
% 
% time11=zeros(4,3);
% time12=zeros(4,3);
% time21=zeros(4,3);
% time22=zeros(4,3);
% 
% cv_error=zeros(4,1);
% cv_error_training=zeros(4,1);
% cv_mse=zeros(4,1);
% avg_time=zeros(4,1);
% trainx=[1:4000;2001:6000;1:2000,4001:6000];% indices for 3-Fold crossvalidation
% valx=[4001:6000;1:2000;2001:4000];
% for n=1:4
%     for e=1:3
%         for l=1:2
%             for m=1:2
%     net1=patternnet(neurons(n));
%     net1.trainFcn = 'traingdm' ;%gradient descent with momentum back propogation
%     net1.divideFcn= 'divideind';  % for Cross validation indices  
%     net1.trainParam.epochs =epochs(e); % maximum No. of epochs in training
%     net1.trainParam.lr = lr(l); % learning rate [0.01 , 0.9]
%     net1.trainParam.mc = momentum(m); % momentum constant [0.1,0.9]
%     for i=1:3; 
% 
%         net1.divideParam.trainInd=[trainx(i,:)];
%         net1.divideParam.valInd=[valx(i,:)];
%         net1.divideParam.testInd=[];
%         tic;
%         [net1,tr, Output]=train(net1,TrainX,Train_dY);
%         time(i)=toc;
%         valX=TrainX(:,valx(i,:));
%         trainingX=TrainX(:,trainx(i,:));
%         y1=net1(valX);
%         y2=net1(trainingX);
%         P_val=single(vec2ind(y1));
%         P_train=single(vec2ind(y2));
%         TrainY_val=TrainY(:,valx(i,:));
%         TrainY_train=TrainY(:,trainx(i,:));
%         error_val(i)=(1- sum((TrainY_val) == P_val) / length(TrainY_val))*100 ;% compare the predicted vs. actual
%         error_train(i)=(1- sum((TrainY_train) == P_train) / length(TrainY_train))*100;
%     end;
% % cv_error(n)=sum(error_val)/size(error_val,1);
% % cv_error_training(n)=sum(error_train)/size(error_train,1);
% % cv_mse(n)=sum(MSE)/size(MSE,1);
% % avg_time(n)=sum(time)/size(time,1);
% % fprintf('avg cv_val_error %f |avg cv_train_error %f for %f neurons\n\n', cv_error(n),cv_error_training(n),neurons(n));
% % fprintf('ave MSE %f for % neurons\n\n', cv_mse(n),neurons(n));
%                 if (l==1) && (m==1)
%                     error11(n,e)=sum(error_val)/size(error_val,1)
%                     time11(n,e)=sum(time)/size(time,1)   
%                     errortrain11(n,e)=sum(error_train)/size(error_train,1)
%                 elseif (l==1) && (m==2)
%                     error12(n,e)=sum(error_val)/size(error_val,1)
%                     time12(n,e)=sum(time)/size(time,1)
%                     errortrain12(n,e)=sum(error_train)/size(error_train,1)
%                 elseif (l==2) && (m==1)
%                     error21(n,e)=sum(error_val)/size(error_val,1)
%                     time21(n,e)=sum(time)/size(time,1) 
%                     errortrain21(n,e)=sum(error_train)/size(error_train,1)
%                      my_net{n,e}=net1
%                 else (l==2) && (m==2)
%                     error22(n,e)=sum(error_val)/size(error_val,1)
%                     time22(n,e)=sum(time)/size(time,1)
%                     errortrain22(n,e)=sum(error_train)/size(error_train,1)
%                     
%                 end;
%             end;
%         end;
%     end;
% end;
% 
% save error11;
% save error12;
% save error21;
% save error22;
% 
% save errortrain11;
% save errortrain12;
% save errortrain21;
% save errortrain22;
% 
% save time11;
% save time12;
% save time21;
% save time22;

%% Final Model MLP
load my_net;% LOAD THE NET ARRAY
tic;
y3=my_net{2,3}(TestX);%USE THE NET WITH BEST VALIDATION RESULT FOR TEST SET
time=toc
P_test=single(vec2ind(y3));%Extracting labels from indicator matrix
TestY(TestY==10)=0;% Converting 10 back to 0 label
P_test(P_test==10)=0;% Converting 10 back to 0 label
error_test=(1- sum((TestY) == P_test) / length(TestY))*100 %Misclassification Error Percentage
C_mat_MLP=confusionmat(TestY,P_test);
figure;
heatmap(C_mat_MLP, unique(P_test), unique(TestY), 1,'Colormap','red','ShowAllTicks',1,'UseLogColorMap',true,'Colorbar',true);
%Plotting a confusion matrix using customized MATLAB code