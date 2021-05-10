function [zernike_mom] = Zernike_Moments(f, ord, repet)
% Function to compute Zernike moments up to order ord.
%       f = Input image M x N matrix (can be grayscale or color)
%       ord = The maximum order (scalar)
%       rep = repetirion  number
%       zernike_mom = Vector containing all the moments up to order ord
%   
% EXAMPLE
%   
%   RGBfile = imgetfile();
%   RGB = imread(RGBfile);
%   if size(RGB, 3)==1
%       gray = RGB;
%   else
%       gray = rgb2gray(RGB);
%   end
%   ord = 4;
%   repet = 2;
%   mom = Zernike_Moments(gray, ord, repet);
%   mom2 = Zernike_Moments(imrotate(gray, 90), ord, repet);
%   mom3 = Zernike_Moments(imresize(gray, 0.5), ord, repet);
%   MOM = [mom;mom2;mom3];
if size(f,3)==3  
    f = rgb2gray(f); 
end

f = double(f);
zernike_mom = [];

for i=0:ord
    for j=0:repet
        if i>=j && mod((i+j),2)==0
            % Computation of Zernike moment with order i and repetition j 
            [~, AOH, ~] = Zernikmoment_MN(f,i,j); 
            zernike_mom = [zernike_mom AOH];
        end
    end
end    

% -------------------------------------------------------------------------

function [Z, A, Phi] = Zernikmoment_MN(f,n,m)
% Function to find the Zernike moments for an M x N grayscale image
%       f = Input image M x N matrix 
%       n = The order of Zernike moment (scalar)
%       m = The repetition number of Zernike moment (scalar)
%       Z = Complex Zernike moment 
%       A = Amplitude of the moment
%       Phi = phase (angle) of the moment (in degrees)
%
[M, N]  = size(f);
y = 1:M; x = 1:N;  % x row, y col

[X,Y] = meshgrid(x,y);
R = sqrt( (2.*X-N-1).^2/N^2 + (2.*Y-M-1).^2/M^2 );
Theta = atan2((M-1-2.*Y+2)/M,(2.*X-N+1-2)/N);

R = (R<=1).*R;
Rad = radialpoly(R,n,m);    % get the radial polynomial
Product = f.*Rad.*exp(-1i*m*Theta);
Z = sum(Product(:));        % calculate the moments
 
cnt = nnz(R)+1;             % count the number of pixels inside the unit circle
Z = (n+1)*Z/cnt;            % normalize the amplitude of moments
A = abs(Z);                 % calculate the amplitude of the moment
Phi = angle(Z)*180/pi;      % calculate the phase of the moment (in degrees)

% -------------------------------------------------------------------------

function rad = radialpoly(r,n,m)
% Function to compute Zernike Polynomials:
%       r = radius
%       n = the order of Zernike polynomial
%       m = the repetition of Zernike moment
rad = zeros(size(r));                     % Initilization
for s = 0:(n-abs(m))/2
  c = (-1)^s*factorial(n-s)/(factorial(s)*factorial((n+abs(m))/2-s)*...
      factorial((n-abs(m))/2-s));
  rad = rad + c*r.^(n-2*s);
end
