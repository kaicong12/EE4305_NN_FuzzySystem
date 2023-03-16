clc;
clearvars; 

[avg_components, downscaled_data, targets, n_train] = load_data();

hidden_sizes = [avg_components, round(avg_components*0.5), round(avg_components*0.75)];
batch_sizes = [10, 20, 50, 100];
regularization_ratios = [0.2, 0.4, 0.6];

for hidden_size = hidden_sizes
    for batch_size = batch_sizes
        for regularization_ratio = regularization_ratios
            epoch_test = [0:batch_size:500];
            fprintf("Training begins for %d batch size and %d hidden neurons and %d regularization ratio\n", batch_size, hidden_size, regularization_ratio);
            [accu_train, accu_val] = train_batch(downscaled_data, targets, batch_size, 500, n_train, hidden_size, regularization_ratio);
            
            [best_train_accuracy, best_train_idx] = max(accu_train);
            [best_val_accuracy, best_val_idx] = max(accu_val);
            fprintf("Best Train Accuracy: %d, Happens at epoch: %d \n", best_train_accuracy*100, epoch_test(best_train_idx));
            fprintf("Best Test Accuracy: %d, Happens at epoch: %d \n", best_val_accuracy*100, epoch_test(best_val_idx));
        end
    end
end



function [accu_train, accu_val] = train_batch(data, labels, batch_size, n_epochs, n_train, n_hidden_size, regularization_ratio)
    epoch_test = [0:batch_size:n_epochs];
    accu_train = zeros(1, length(epoch_test));
    accu_val = zeros(1, length(epoch_test));

    % train_test split data
    training_data = data(:, 1:n_train);
    training_label = labels(1:n_train);
    validation_data = data(:, n_train+1:size(data, 2));
    validation_label = labels(n_train+1:length(labels));

    for i = 1:length(epoch_test)
        net = patternnet(n_hidden_size);
        net.trainParam.lr = 0.001;
        net.trainParam.showWindow = false;
        net.trainParam.epochs=epoch_test(i);
        net.performParam.regularization = regularization_ratio;

        net = train(net, training_data, training_label);
        train_out = net(training_data);
        accu_train(i) = 1-mean(abs(train_out - training_label));

        val_out = net(validation_data);    
        accu_val(i) = 1- mean(abs(val_out - validation_label)); 
    end

%     plot(epoch_test,  accu_train, epoch_test, accu_val);
%     xlabel("epochs");
%     ylabel("accuracy");
%     filename = sprintf("%d batchSize %d neurons tansig activation", batch_size, n_hidden_size);
%     title(filename);
%     legend({"training", "testing"}, 'Location', 'northeast');
%     saveas(gcf, filename, 'png');
end

function [avg_components, downscaled_data, targets, n_train] = load_data()
%     avg_components = get_average_components();
    avg_components = 62;
    [downscaled_data, targets, n_train] = extract_top_components(avg_components);
    downscaled_data = downscaled_data(:, 2:size(downscaled_data, 2));
    targets = targets(:, 2:size(targets, 2));
end

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