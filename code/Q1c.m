weights = -1 + (1+1) .* rand(2,1);  % generate random weights between (-1, 1)
initial_weights = weights;  % record the initial weight

% training params
n_iters = 1e6;
threshold = 1e-10;

% create matrix to record the changes in parameters during training
x = zeros();
y = zeros();
errors = zeros();

i = 1;
earlyStopping = false;
while i <= n_iters
    % training loop
    X_prev = weights;
    x(i) = weights(1);
    y(i) = weights(2);

    gradient = rosenGrad(weights(1), weights(2));
    heissen = rosenHess(weights(1), weights(2));
    
    % replace learning rate here
    weights = weights - (inv(heissen) * gradient);
    
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
plot(iterations, x, iterations, y);
xlabel('iterations');
ylabel('value');
legend({'x', 'y'}, 'Location', 'northeast');


function Df = rosenGrad(x, y)
    k = [x-1; y-(x^2)];
    Df = [2*k(1)-(400*x*k(2)); 200*k(2)];
end

function H = rosenHess(x, y)
    df2dx2 = 2 - 400*y + 1200*(x^2);
    df2dy2 = 200;
    df2dxdy = -400 * x;
    H = [df2dx2, df2dxdy; df2dxdy, df2dy2];
end