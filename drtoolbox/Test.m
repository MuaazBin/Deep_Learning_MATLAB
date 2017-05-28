load mnist.mat
Xtrain=[train0;train1;train2;train3;train4;train5;train6;train7;train8;train9];
Y=[5923,6742,5958,6131,5842,5421,5918,6265,5851,5949];
Z=sum(Y)
X0=ones(5923,1)*0
X1=ones(6742,1)*1
X2=ones(5958,1)*2
X3=ones(6131,1)*3
X4=ones(5842,1)*4
X5=ones(5421,1)*5
X6=ones(5918,1)*6
X7=ones(6265,1)*7
X8=ones(5851,1)*8
X9=ones(5949,1)*9
Ytrain=[X0;X1;X2;X3;X4;X5;X6;X7;X8;X9]
