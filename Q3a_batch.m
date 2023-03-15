clear all;
clc;

% read in image and label data
train_path = "group_1/train";
test_path = "group_1/test";
[X_train, y_train] = load_data(train_path);
[X_test, y_test] = load_data(test_path);


epochs = 500;
net = perceptron;
net.inputWeights{1,1}.learnParam.lr = 0.001;
net.biases{1,1}.learnParam.lr = 0.001;
net.trainParam.showWindow = false;
net.trainParam.epochs = epochs;

% 7:3 split for validation set
n_sample = size(X_train, 2);
upperQuartile = round(n_sample*0.75);
net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:upperQuartile;
net.divideParam.valInd = upperQuartile+1:n_sample;

% shuffle_X = X_train(:, randperm(size(X_train, 2)));
[net, tr] = train(net, X_train, y_train);

f1 = figure;
f1.Position = [100 100 700 500];
plot(tr.epoch, 1-tr.perf, tr.epoch, 1-tr.vperf);
xlabel('Epoch');
ylabel('Accuracy (%)');
title('Training in Batch mode');
legend({'training', 'test'}, 'Location', 'northwest');


function [images, labels] = load_data(directory)
    dir_struct = dir(directory);
    % it is given that image size is 256*256
    images = zeros([256*256, length(dir_struct)]);
    labels = zeros();
    
    for i = 1:length(dir_struct)
        % Skip over the directories '.' and '..'
        if strcmp(dir_struct(i).name,'.') || strcmp(dir_struct(i).name,'..')
            continue
        end
        
        % image matrix
        file_path = fullfile(directory, dir_struct(i).name);
        I = imread(file_path);
        V = I(:);
        images(:,i) = V;
    
        % label for this image
        tmp = strsplit(file_path, {'_', '.'});
        labels(i)= str2num(tmp{3});
    end
end