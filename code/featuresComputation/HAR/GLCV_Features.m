function [F] = GLCV_Features(GLCVS, normalization)
% GLCV_Features calculates features from each GLCVS in input
% Example
%       I = imread('cell.tif');
%       [~, GLCVS] = GLCoOcc(I); 
%       [F] = GLCV_Features(GLCVS);
if nargin == 1
    normalization = false;
end

n = size(GLCVS,2);
F = zeros(20,n);

for k = 1:n
    G = double(GLCVS{k});
    NL = G(end,1);% Gray level number
    G(end,:)=[];
    G(:,3) = G(:,3)/(sum(G(:,3)));% Normalize the GLCV in question
    
    xy = G(:,1) + G(:,2);
    x_y = G(:,1) - G(:,2);
    dissi = abs(x_y);
    contr = dissi.^2;

    px = accumarray(G(:,1),G(:,3));
    py = accumarray(G(:,2),G(:,3));
    pxy=accumarray(xy-1,G(:,3));
    px_y=accumarray(dissi+1,G(:,3));   
    
    HX = - sum(px(:).*log(px(:) + eps));
    HY = - sum(py(:).*log(py(:) + eps));
    HXY = - sum( G(:,3).*log(px(G(:,1)).*py(G(:,2)) + eps));         %%%%HXY1 e HXY2 sono uguali per matrici normalizzate 
    
    u_x = sum(G(:,1).*G(:,3));
    u_y = sum(G(:,2).*G(:,3));
    u = (u_x + u_y)/2;
    s_xy = (sum(((G(:,1) - u_x).^2).*G(:,3))  * sum(((G(:,2) - u_y).^2).*G(:,3)))^0.5;
    xy_u = xy - u_x - u_y;
        
    F(1,k) = sum(G(:,3).^2); % 1 Angular Second Moment = Energy(Matlab) = Uniformity [1,2] 
    F(2,k) = sum(contr.*G(:,3)); % 2 Contrast [1,2] 
    F(3,k) = (sum(G(:,1).*G(:,2).*G(:,3)) - u_x*u_y)/(s_xy+eps);  % 3 Correlation [1,2]
    F(4,k) = sum(G(:,3).*((G(:,1) - u).^2));  % 4 Sum of squares: Variance [1] 
    F(5,k) = sum(G(:,3)./( 1 + contr)); % 5 Inverse Difference Moment = Homogeneity [2] in MATLAB sum(G(:,3)./( 1 + dissi)); 
    F(6,k) = sum((xy).*G(:,3)); % 6 Sum average [1]      
    F(7,k) = sum((((xy) - F(6,k)).^2).*G(:,3)); % 7 Sum variance [1]     
    F(8,k) = - sum(pxy(:).*log(pxy(:) + eps)); % 8 Sum entropy [1] 
    F(9,k) = - sum((G(:,3).*log(double(G(:,3)) + eps)));   % 9 Entropy [2]  
    
    F(14,k) = sum(dissi.*G(:,3)); % 14 Difference Average = Dissimilarity [2] 
    F(10,k) = sum(((dissi- F(14,k)).^2).*G(:,3)); % 10 Difference variance [1]
    F(11,k) = - sum(px_y(:).*log(px_y(:) + eps));  % 11 Difference entropy [1] 
    F(12,k) = ( F(9,k) - HXY)/ ( max([HX,HY])) ;  % 12 Information measure of correlation1 [1] 
    F(13,k) = ( 1 - exp( -2*((HXY - F(9,k)))))^0.5;  % 13 Informaiton measure of correlation2 [1]    

    F(15,k) = sum(G(:,1).*G(:,2).*G(:,3)); % 15 Autocorrelation [2] 
    F(16,k) = max(G(:,3)); % 16 Maximum probability [2]
    F(17,k) = sum((xy_u.^3).*G(:,3)); % 17 Cluster Shade [2] 
    F(18,k) = sum((xy_u.^4).*G(:,3));  % 18 Cluster Prominence [2] 
    F(19,k) = sum(G(:,3)./ (1 + dissi./NL)); % 19 Inverse difference normalized (INN) [3]        
    F(20,k) = sum(G(:,3)./ (1 + contr./(NL*NL)));% 20 Inverse difference moment normalized [3] 
end
% 1. R. M. Haralick, K. Shanmugam, and I. Dinstein, Textural Features of Image Classification
% 2. L. Soh and C. Tsatsoulis, Texture Analysis of SAR Sea Ice Imagery Using Gray Level Co-Occurrence Matrices
% 3. D A. Clausi, An analysis of co-occurrence texture statistics as a function of grey level quantization

if normalization
    for k = 1:n
        F(2,k) = (1/((NL-1)^2))*F(2,k);       
        F(4,k) = F(4,k)/(NL*NL);  % 4 Sum of squares: Variance [1]    
        F(6,k) = F(6,k)/(NL*NL); % 6 Sum average [1]
        F(7,k) = F(7,k)/(NL*NL); % 7 Sum variance [1]
        F(8,k) = (1/(2*log(NL)))*F(8,k); % 8 Sum entropy [1]
        F(9,k) = (1/(2*log(NL)))*F(9,k);
        F(10,k) = F(10,k)/NL; % 10 Difference variance [1]
        F(11,k) = (1/(2*log(NL)))*F(11,k);  % 11 Difference entropy [1]
        F(12,k) = - F(12,k);  % 12 Information measure of correlation1 [1]
        F(14,k) = F(14,k)/NL; % 14 Difference Average = Dissimilarity [2]
        F(15,k) = F(15,k)/(NL*NL); % 15 Autocorrelation [2]
        F(17,k) = -F(17,k)/(NL^3); % 17 Cluster Shade [2]
        F(18,k) = F(18,k)/(NL^4);  % 18 Cluster Prominence [2]
    end
end