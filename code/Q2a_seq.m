clc;
clearvars;

hiddenSizes = [5,6,7,10,11,12,15,20,50];

% training set
f = @(x) 1.2*sin(pi*x) - cos(2.4*pi*x);
X_train = -1:0.05:1;
y_train = f(X_train);

% test set
X_test = -3:0.01:3;
y_test = f(X_test);

for hiddenSize = hiddenSizes
    net = train_seq(hiddenSize, X_train, y_train);
    y_train_pred = net(X_train);
    y_pred = net(X_test);

    % check training accuracy
    plot(X_train, y_train, X_train, y_train_pred);
    filename = sprintf("Training Accuracy %d hidden neurons.png", hiddenSize);
    title(filename);
    xlabel('x');
    ylabel('y');
    legend({'actual', 'predicted'}, 'Location', 'northwest');
    saveas(gcf, filename, 'png');

    % plot out predictions against y_test
    plot(X_test, y_test, X_test, y_pred);
    test_filename = sprintf("Testing Accuracy %d hidden neurons.png", hiddenSize);
    title(test_filename);
    xlabel('x');
    ylabel('y');
    legend({'actual', 'predicted'}, 'Location', 'northwest');
    saveas(gcf, test_filename, 'png');
end

function [net] = train_seq(hiddenSize, train_X, train_X_output)

    x_train = num2cell(train_X);
    x_train_label = num2cell(train_X_output);
    
    net = fitnet(hiddenSize, 'traingda'); % traingda appears to give the best outcomes.
    
    net.divideParam.trainRatio = 1;
    net.divideParam.valRatio = 0;
    net.divideParam.testRatio = 0;
    net.trainParam.lr = 0.001;
    
    for i=1:500
        idx = randperm(length(x_train));
        net = adapt(net, x_train(:, idx), x_train_label(:, idx));      
    end
end