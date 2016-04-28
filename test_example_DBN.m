function test_example_DBN
load ../data/mnist_40000_10000;
addpath('../DBN');
addpath('../NN');
addpath('../util');
train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

rand('state',0)
//train dbn
dbn.sizes = [100 200]; //DBN的结构，v1层为raw pixel/原始图片，h1/v2层的节点数为100，h2/v3层的节点数为200
opts.numepochs =   3;
opts.batchsize = 100;
opts.momentum  =   0; //记录以前的更新方向，并与现在的方向结合下，从而加快学习的速度
opts.alpha     =   1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 10);
nn.activation_function = 'sigm';

//train nn
//得到DBN的初始化参数后，用nn进行微调
opts.numepochs =  3;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);

assert(er < 0.10, 'Too big error');
