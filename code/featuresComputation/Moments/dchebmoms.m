function [M,Mc,P1,P2] = dchebmoms(F,ord)
%DCHEBMOMS discrete Chebyshev moments of an image
%   M=dchebmoms(F,ord) computes the matrix M of the discrete Chebyshev moments
%   of the image F, up to order ord.
%   [M,Mc]=dchebmoms(F,ord) computes also the continuous Chebyshev moments Mc.
%   [M,Md,P1,P2]=dchebmoms(F,ord) returns the evaluation of orthogonal
%   polynomials on both axes.

n = ord;
type = 'DChebyshev';
usesimpson = 1 && (nargin>1);	% use Simpson to compute integrals

[m1 m2] = size(F);
if usesimpson && ~(2*round(m1/2)-m1), m1 = m1-1; end	% for Simpson
if usesimpson && ~(2*round(m2/2)-m2), m2 = m2-1; end	% for Simpson
F = double(F(1:m1,1:m2));
F = mat2gray(F);
x = [0:m1-1]';
y = [0:m2-1]';

[alfa1 beta1] = opcoef(type,n,m1);	% recursion coefficients
[alfa2 beta2] = opcoef(type,n,m2);	% recursion coefficients
P1 = opevmat(alfa1,beta1,x);		% values of polynomials on x
P2 = opevmat(alfa2,beta2,y);		% values of polynomials on y

% moments
M = P1'*F*P2;
if nargout>1, Mc = opcmoms(F,P1,P2,usesimpson); end

