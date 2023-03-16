clearvars;
close all;

img_gray = imread('group_1/train/0001_0_highway.jpg');
X = double(img_gray);
[coeff, score, latent] = pca(X);


% Plot the scree plot
figure;
variance_explained = cumsum(latent)./sum(latent);
plot(variance_explained);
xlabel('Number of Principal Components');
ylabel('Cumulative Variance Explained');
title('Scree Plot');


for k = [10, 15, 20, 50, 100, 200]
    % Select the number of components to retain
    Uk = coeff(:, 1:k);
    Sk = diag(latent(1:k));

    % Reconstruct the image using the selected components
    X_approx = score(:, 1:k) * Uk' + mean(X);
    img_approx = uint8(X_approx);

    % Display the downsized image
    figure;
    imshow(img_approx);
    title(sprintf('Downsized Image (%d components)', k));
end