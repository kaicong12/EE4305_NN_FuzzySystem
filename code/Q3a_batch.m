clearvars;
close all;

% read in all images (both train and test) as data
% split train and test set later on
[data, target, split_idx] = load_data();

epochs = 500;
net = perceptron;
net.inputWeights{1,1}.learnParam.lr = 0.001;
net.biases{1,1}.learnParam.lr = 0.001;
net.trainParam.showWindow = false;
net.trainParam.epochs = epochs;

% split the datase at split_idx as test set
n_sample = size(data, 2);
net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:split_idx;
net.divideParam.testInd = split_idx+1:n_sample;

% shuffle_X = X_train(:, randperm(size(X_train, 2)));
[net, tr] = train(net, data, target);

f1 = figure;
f1.Position = [100 100 700 500];
plot(tr.epoch, 1-tr.perf, tr.epoch, 1-tr.tperf);
xlabel('Epoch');
ylabel('Accuracy (%)');
title('Training in Batch mode');
legend({'training', 'test'}, 'Location', 'northwest');


function [images, labels] = read_image(directory)
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

    images = images(:, 3:size(images, 2));
    labels = labels(3: length(labels));
end

function [data, target, test_idx] = load_data()
    % read in image and label data
    train_path = "group_1/train";
    test_path = "group_1/test";
    [X_train, y_train] = read_image(train_path);
    [X_test, y_test] = read_image(test_path);
    test_idx = size(X_train, 2);
    
    data = cat(2, X_train, X_test);
    target = cat(2, y_train, y_test);
end
