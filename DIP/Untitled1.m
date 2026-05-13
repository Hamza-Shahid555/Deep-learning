

% Read image
img = imread('pout.tif');  % built-in MATLAB image

% Apply histogram equalization
img_eq = histeq(img);

% Display results
figure;
subplot(2,2,1); imshow(img);        title('Original Image');
subplot(2,2,2); imshow(img_eq);     title('Equalized Image');
subplot(2,2,3); imhist(img);        title('Original Histogram');
subplot(2,2,4); imhist(img_eq);     title('Equalized Histogram');