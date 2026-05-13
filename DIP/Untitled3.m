% Part 3: Histogram Matching
clc; clear all; close all;

% Source and reference images
src = imread('pout.tif');
ref = imread('cameraman.tif');   % another built-in MATLAB image

% Apply histogram matching
matched = imhistmatch(src, ref);

% Display
figure;
subplot(2,3,1); imshow(src);      title('Source Image');
subplot(2,3,2); imshow(ref);      title('Reference Image');
subplot(2,3,3); imshow(matched);  title('Matched Image');
subplot(2,3,4); imhist(src);      title('Source Histogram');
subplot(2,3,5); imhist(ref);      title('Reference Histogram');
subplot(2,3,6); imhist(matched);  title('Matched Histogram');