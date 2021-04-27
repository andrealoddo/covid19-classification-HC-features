function [GLCMS, GLCVS] = GLCoOcc(I1, varargin)
% GLCM calculates the GLCM matrix for two given images using a specific 
% offset. If [0,0] is specified it computes the summation GLCM 
% Example
%       I = imread('cell.tif');
%       [GLCMS] = GLCoOcc(I, [0 1]); 
%
% Example2
%       RGB = imread('RGBcell.tif');
%       [~, GLCVS] = GLCoOcc(RGB(:,:,1), RGB(:,:,2)); 
p = inputParser;
addRequired(p,'I1'); % Required image
addOptional(p,'I2', I1, @(x) size(x,1)==size(I1,1) && size(x,2)==size(I1,2)); % check images, if I1 == I2 use only one bands (classical GLCM)
addParameter(p,'offsets', [0 1;-1 1;-1 0;-1 -1], @(x) isnumeric(x) && size(x,2) == 2); % check offsets
addParameter(p,'graylevel',0, @(x) isnumeric(x) && isscalar(x) && (x > 0)); % Default graylevel numbers (number of gray level inside the image)
addParameter(p,'symmetric', true, @(x) islogical(x)); % Default symmetric
parse(p,I1,varargin{:});

I2 = p.Results.I2;
I1 = doubleimg(checkChannel(I1));%convert to double and check color
I2 = doubleimg(checkChannel(I2));%convert to double and check color
offsets = p.Results.offsets;
symmetric = p.Results.symmetric;
if p.Results.graylevel == 0
    NL =  double(max(max(I1(:)), max(I2(:))))+1;
else
    NL =  p.Results.graylevel;
end

% Compute GLCMS
numOffsets = size(offsets,1);
GLCMS = zeros(NL,NL,numOffsets);

for k = 1 : numOffsets
    GLCMS(:,:,k) = computeGLCM(I1, I2, offsets(k,:),NL);
    if symmetric
        GLCMS(:,:,k) = GLCMS(:,:,k) + GLCMS(:,:,k).'; % Reflect glcm across the diagonal
    end
end

if nargout==2 % If asked computes GLCVS
    GLCVS = cell(1, numOffsets);
    for k = 1 : numOffsets
        [row,col,v] = find(GLCMS(:,:,k));
        GLCVS{k} = [row,col,v;NL,NL,numOffsets];
    end
end

%--------------------------------------------------------------------------
function img = checkChannel(img)% converts into double
%If more than one channel convert to gray
if size(img, 3) > 1
    img = rgb2gray(img);
end

%--------------------------------------------------------------------------
function img = doubleimg(img)% converts into double

if isfloat(img)
    if max(img(:)) <= 1
        img = img*255;
    end
else
    img = double(img);
end

%--------------------------------------------------------------------------
function oneGLCM = computeGLCM(a, b, offset, nl) % computes GLCM given one offset

[a, b] = shiftMatrices(a, b,offset); % Shift the second image using the offset
a2 = a(:)+1; % Vectorises image and scales gray level 
b2 = b(:)+1; % Vectorises image and scales gray level 
oneGLCM = accumarray([a2, b2], 1, [nl nl]);