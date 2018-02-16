clc;
clear all;
close all;
addpath('C:\Others\SEM-1\FSL\Data Sets\Faces in the Wild\faceData\')
imgFolder=('C:\Others\SEM-1\FSL\Data Sets\Faces in the Wild\faceData\2002\07\21');
imgSetVector = imageSet(imgFolder,'recursive');
A=[];
for k=1:36
h=imshow(read(imgSetVector,k));
Faces{k} = getimage(h);
end
Faces=cell2mat(Faces);
Faces = im2double(Faces);
Faces = rgb2gray(Faces);

autoEncFaces = trainAutoencoder(transpose(Faces),784,'MaxEpochs',100);
encodedFaces = encode(autoEncFaces,transpose(Faces));
reconstructedFaces = predict(autoEncFaces, transpose(Faces));
diff = 0;
for i = 1:3096
    for j = 1:86
        diff = diff + ((Faces(j,i) - reconstructedFaces(i,j))^2);
    end
end
errMseFaces = diff/(86*3096);
psnrErrorFaces = psnr(Faces,reconstructedFaces);
x=reconstructedFaces';
B_Show=reshape(x,[86 36 86]);
B_to_disp=reshape(B_Show,[86*86 36]);
figure(1);
show(B_to_disp);