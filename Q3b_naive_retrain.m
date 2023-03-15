clearvars;

for scale = [0.5, 0.25, 0.125]
    [scaled_data, targets, n_train] = load_data(scale);
    scaled_data = scaled_data(:, 2:size(scaled_data, 2));
    targets = targets(:, 2:size(targets, 2));

    epochs = 500;
    net = perceptron;
    net.inputWeights{1,1}.learnParam.lr = 0.001;
    net.biases{1,1}.learnParam.lr = 0.001;
    net.trainParam.showWindow = false;
    net.trainParam.epochs = epochs;
    
    n_sample = size(scaled_data, 2);
    net.divideFcn = 'divideind';
    net.divideParam.trainInd = 1:n_train;
    net.divideParam.testInd = n_train+1:n_sample;
    
    % shuffle_X = X_train(:, randperm(size(X_train, 2)));
    [net, tr] = train(net, scaled_data, targets);
    
    f1 = figure;
    f1.Position = [100 100 700 500];
    plot(tr.epoch, 1-tr.perf, tr.epoch, 1-tr.tperf);
    xlabel('Epoch');
    ylabel('Accuracy (%)');
    dimension = round(256*scale);
    title(sprintf('Training with %d x %d image', dimension, dimension));
    legend({'training', 'test'}, 'Location', 'northwest');
end



function [downscaled_data, targets, n_train] = load_data(scale)
    
    new_dimension = round(256*scale);
    directories = ["group_1/train" "group_1/test"];
    downscaled_data = zeros(new_dimension^2, 1);
    targets = zeros();

    for directory = directories
        dir_struct = dir(directory);
        n_sample = length(dir_struct);
        if directory == "group_1/train"
            n_train = n_sample;
        end
        
        % it is known that the images are of dimension 256 x 256
        data = zeros([new_dimension^2, n_sample]);
        labels = zeros();
    
        for i=1:n_sample
            % Skip over the directories '.' and '..'
            if strcmp(dir_struct(i).name,'.') || strcmp(dir_struct(i).name,'..')
                continue
            end
            
            % image matrix
            file_path = fullfile(dir_struct(i).folder, dir_struct(i).name);
            img_gray = imread(file_path);
            img_resized = imresize(img_gray, scale);
            V = img_resized(:);
            data(:,i) = V;
    
            % label for this image
            tmp = strsplit(file_path, {'_', '.'});
            labels(i)= str2num(tmp{3});
        end

        downscaled_data = [downscaled_data data(:, 3:size(data, 2))];
        targets = [targets labels(3: length(labels))];
    end
end