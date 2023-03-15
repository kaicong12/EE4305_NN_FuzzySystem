hiddenSizes = [1,2,3,4,5,6,7,10,11,12,15,20,50];

% training set
f = @(x) 1.2*sin(pi*x) - cos(2.4*pi*x);
X_train = -1:0.05:1;
y_train = f(X_train);

% test set
X_test = -3:0.01:3;
y_test = f(X_test);


trainFcn = 'trainlm';  % toggle between 'trainlm' and 'trainbr'
trainMode = 'trainb';  % toggle between 'trains' (Sequential) and 'trainb' (Batch Mode)
for i = 1:1
    hiddenSize = hiddenSizes(i);
    net = fitnet(hiddenSize, trainFcn);

    % net.trainFcn = trainMode;
    net.trainParam.epochs = 500;
    net.trainParam.showWindow = false;
    net.trainParam.lr = 0.005;


    net = train(net,X_train,y_train);
    y_train_pred = net(X_train);
    y_pred = net(X_test);


    % check training accuracy
    f1 = figure;
    f1.Position = [100 100 600 400];
    plot(X_train, y_train, X_train, y_train_pred);
    title(sprintf("Training performance with %d hidden neurons", hiddenSize));
    xlabel('x');
    ylabel('y');
    legend({'actual', 'predicted'}, 'Location', 'northwest');

    % plot out predictions against y_test
    f2 = figure;
    f2.Position = [100 100 600 400];
    plot(X_test, y_test, X_test, y_pred);
    title(sprintf("Testing performance with %d hidden neurons", hiddenSize));
    xlabel('x');
    ylabel('y');
    legend({'actual', 'predicted'}, 'Location', 'northwest');
end