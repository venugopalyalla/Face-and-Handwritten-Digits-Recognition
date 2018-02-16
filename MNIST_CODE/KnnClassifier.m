trainLabels = loadMNISTLabels('train-labels.idx1-ubyte');
testLabels = loadMNISTLabels('t10k-labels.idx1-ubyte');
mdl = fitcknn(transpose(encodedImages),trainLabels);
knnAccr(1:5,1) = 0;
numK = 30;
for k = 1:15
    mdl.NumNeighbors = numK;
    knnPredictions = predict (mdl,transpose(encodedTestImages));
    count = 0;
    for z = 1:10000
        if(knnPredictions(z,1) == testLabels(z,1))
            count = count + 1;
        end
    end
    knnAccr(k,1) = (count/10000) * 100;
    numK = numK + 10;
end
x = [30,40,50,60,70,80,90,100,110,120,130,140,150,160,170];
y = knnAccr(:,1);
figure(1);
plot(x,y);
axis([0,200,90,95]);