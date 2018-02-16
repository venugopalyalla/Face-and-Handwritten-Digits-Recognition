clear;
images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
testLabels = loadMNISTLabels('t10k-labels.idx1-ubyte');
testImages = loadMNISTImages('t10k-images.idx3-ubyte');
autoenc1 = trainAutoencoder(images,85,'MaxEpochs',100);
encodedImages = encode(autoenc1,images);
encodedTestImages = encode(autoenc1,testImages);
reconstructedImages = predict(autoenc1, images);
diff = 0;
for i = 1:784
    for j = 1:60000
        diff = diff + ((images(i,j) - reconstructedImages(i,j))^2);
    end
end
errMse = diff/(784*60000);
psnrError = psnr(images,reconstructedImages);
disp(errMse);
disp(psnrError);
show(reconstructedImages(:,1:784));