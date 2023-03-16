img = imread('group_1/train/0001_0_highway.jpg');
figure;
imshow(img);

scales = [0.5, 0.25, 0.125];

for s = scales;
    img_resized = imresize(img, s);
    
    % Display the downsized image
    figure;
    imshow(img_resized);
    title(sprintf("Downsized dimension %d x %d", 256*s, 256*s));
end
