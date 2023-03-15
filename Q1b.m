weights = -1 + (1+1) .* rand(2,1);  % generate random weights between (-1, 1)
initial_weights = weights;  % record the initial weight

% training params
n_iters = 1e6;
learning_rate_low = 0.001;  
learning_rate_high = 1.0;
threshold = 1e-10;

% create matrix to record the changes in parameters during training
x = zeros();
y = zeros();
gradientX = zeros();
gradientY = zeros();
errors = zeros();


i = 1;
earlyStopping = false;
lr = learning_rate_high;
while i <= n_iters
    % training loop
    X_prev = weights;
    x(i) = weights(1);
    y(i) = weights(2);
    gradient = rosenGrad(weights(1), weights(2));
    gradientX(i) = gradient(1);
    gradientY(i) = gradient(2);
    
    % replace learning rate here
    weights = weights - (lr * gradient);
    
    % stop training if delta weights are lesser than a threshold
    error = norm(weights - X_prev);  % euclidean distance 
    errors(i) = error;
    if error < threshold
        earlyStopping = true;
        break
    end

    i = i + 1;
end


% plot results (trajectory in 2D space)
figure;
x_values = [-2:0.01:2]; 
y_values = [-2:0.01:2];
f = @(x,y) (1-x).^2 + 100*(y-x.^2).^2;
[xx, yy] = meshgrid(x_values, y_values);
ff = f(xx,yy);
fn_output = f(x, y);
contour(xx, yy, ff, 100);

hold on;
scatter(x, y);
xlabel('x');
ylabel('y');
hold off;

% function values as it approaches the global minimum
figure;
if earlyStopping == true
    iterations = [1: i];
else
    iterations = [1: i-1];
end


plot(iterations, fn_output); 
xlabel('iterations');
ylabel('fn_output');

figure;
plot(iterations, gradientX);
xlabel('iterations');
ylabel('gradientX');

figure;
plot(iterations, gradientY);
xlabel('iterations');
ylabel('gradientY');

function gradient = rosenGrad(x, y)
    k = [x-1; y-(x^2)];
    gradient = [2*k(1)-(400*x*k(2)); 200*k(2)];
end