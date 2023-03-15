% read in image and label data
train_path = "group_1/train";
test_path = "group_1/test";
[X_train, y_train] = load_data(train_path);
[X_test, y_test] = load_data(test_path);



% Experiment with multiple epochs
epochs_range = [1000];
for i=1:length(epochs_range)
    cur_epoch = epochs_range(i);
    [net, accu_train, accu_test] = train(X_train, y_train, X_test, y_test, cur_epoch);

    % Plot training results
    f1 = figure;
    f1.Position = [100 100 700 500];
    plot(1:cur_epoch, accu_train, 1:cur_epoch, accu_test);
    xlabel("Epoch");
    ylabel("Accuracy (%)");
    title(sprintf("Sequential Training with %d epochs", cur_epoch))
    legend({'training', 'test'}, 'Location', 'northwest');
end



function [net, accu_train, accu_test] = train(X_train, y_train, X_test, y_test, cur_epoch)
    net = perceptron;
    net.trainParam.epochs = cur_epoch;
    
    accu_train = zeros(1, cur_epoch);
    accu_test = zeros(1, cur_epoch);
    
    % Sequential Training loop
    for i=1:cur_epoch
        idx = randperm(size(X_train, 2));
        [net,a,e] = adapt(net, X_train(:,idx), y_train(:,idx));
    
        pred_train = net(X_train(:,idx));
        accu_train(i) = 1 - mean(abs(pred_train - y_train(:,idx)));
    
        pred_test = net(X_test);
        accu_test(i) = 1 - mean(abs(pred_test - y_test));
    end
end


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