net = feedforwardnet(25);
targets = transpose(labels);
addTargets(:,:) = 0;
for y = 1:60000
    addTargets(1,y) = targets(1,y) + 1;
end
trainOuputs = full(ind2vec(addTargets));
net = train(net,encodedImages,trainOuputs);
testIp = encodedTestImages;
testOp = net(testIp);
vec_testOp = vec2ind(testOp);
count = 0;
for i = 1:numel(vec_testOp)
    if(vec_testOp(i) == (transpose(testLabels(i)) + 1))
        count = count + 1;
    end
end
accuracy = (count/numel(vec_testOp)) * 100;
disp(accuracy);