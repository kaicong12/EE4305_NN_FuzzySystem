clearvars;

% get average components for the entire dataset
% perform PCA on each image using that n_components
avg_components = get_average_components();
[downscaled_data, targets, n_train] = extract_top_components(avg_components);
downscaled_data = downscaled_data(:, 2:size(downscaled_data, 2));
targets = targets(:, 2:size(targets, 2));

epochs = 200;
net = perceptron;
net.inputWeights{1,1}.learnParam.lr = 0.001;
net.biases{1,1}.learnParam.lr = 0.001;
net.trainParam.showWindow = false;
net.trainParam.epochs = epochs;

n_sample = size(downscaled_data, 2);
net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:n_train;
net.divideParam.testInd = n_train+1:n_sample;

% shuffle_X = X_train(:, randperm(size(X_train, 2)));
[net, tr] = train(net, downscaled_data, targets);

f1 = figure;
f1.Position = [100 100 700 500];
plot(tr.epoch, 1-tr.perf, tr.epoch, 1-tr.tperf);
xlabel('Epoch');
ylabel('Accuracy (%)');
title('Training in Batch mode');
legend({'training', 'test'}, 'Location', 'northwest');


function avg_components = get_average_components()

    directories = ["group_1/train" "group_1/test"];
    components = zeros();
    
    offset = 0;
    for directory = directories
        dir_struct = dir(directory);
        for i = 1:length(dir_struct)
            % Skip over the directories '.' and '..'
            if strcmp(dir_struct(i).name,'.') || strcmp(dir_struct(i).name,'..')
                continue
            end
            
            % image matrix
            file_path = fullfile(dir_struct(i).folder, dir_struct(i).name);
            img_gray = imread(file_path);
            X = double(img_gray);
            [coeff, score, latent] = pca(X);
    
            variance_explained = cumsum(latent)./sum(latent);
            num_components = find(variance_explained >= 0.99, 1, 'first');
            components(offset+i) = num_components;
        end
        offset = offset + length(dir_struct);
    end
    

    avg_components = ceil(mean(components));
end


function [downscaled_data, targets, n_train] = extract_top_components(n_components)

    directories = ["group_1/train" "group_1/test"];
    downscaled_data = zeros(256*256, 1);
    targets = zeros();

    for directory = directories
        dir_struct = dir(directory);
        n_sample = length(dir_struct);
        if directory == "group_1/train"
            n_train = n_sample;
        end
        
        % it is known that the images are of dimension 256 x 256
        data = zeros([256*256, n_sample]);
        labels = zeros();
    
        for i=1:n_sample
            % Skip over the directories '.' and '..'
            if strcmp(dir_struct(i).name,'.') || strcmp(dir_struct(i).name,'..')
                continue
            end
            
            % image matrix
            file_path = fullfile(dir_struct(i).folder, dir_struct(i).name);
            img_gray = imread(file_path);
            X = double(img_gray);
            [coeff, score, latent] = pca(X);
            % Select the number of components to retain
            Uk = coeff(:, 1:n_components);
            Sk = diag(latent(1:n_components));
        
            % Reconstruct the image using the selected components
            X_approx = score(:, 1:n_components) * Uk' + mean(X);
            img_approx = uint8(X_approx);
            data(:,i) = img_approx(:);
    
            % label for this image
            tmp = strsplit(file_path, {'_', '.'});
            labels(i)= str2num(tmp{3});
        end

        downscaled_data = [downscaled_data data(:, 3:size(data, 2))];
        targets = [targets labels(3: length(labels))];
    end
end
