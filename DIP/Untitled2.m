% Part 2: Manual Histogram Equalization
clc; clear all; close all;

img = imread('pout.tif');
img = im2double(img);         % convert to 0-1 range
[r, c] = size(img);
total_pixels = r * c;

% Step 1: Compute histogram (256 bins)
hist_count = zeros(1, 256);
img_int = im2uint8(img);      % work with 0-255 values

for i = 1:r
    for j = 1:c
        val = img_int(i,j) + 1;   % MATLAB index starts at 1
        hist_count(val) = hist_count(val) + 1;
    end
end

% Step 2: Compute CDF
cdf = cumsum(hist_count);

% Step 3: Normalize CDF to get mapping
cdf_min = min(cdf(cdf > 0));
mapping = round(((cdf - cdf_min) / (total_pixels - cdf_min)) * 255);

% Step 4: Apply mapping to image
img_out = zeros(r, c, 'uint8');
for i = 1:r
    for j = 1:c
        img_out(i,j) = mapping(img_int(i,j) + 1);
    end
end

% Display
figure;
subplot(2,2,1); imshow(img_int);   title('Original Image');
subplot(2,2,2); imshow(img_out);   title('Equalized (Manual)');
subplot(2,2,3); imhist(img_int);   title('Original Histogram');
subplot(2,2,4); imhist(img_out);   title('Equalized Histogram');