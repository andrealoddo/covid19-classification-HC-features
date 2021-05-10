function [M,Md,PP1,PP2] = cheb2moms(F,ord)
%CHEB2MOMS second kind Chebyshev moments of an image
%   M=cheb2moms(F,ord) computes the matrix M the continuous Chebyshev moments
%   of the image F, up to order ord.
%   [M,Md]=cheb2moms(F,ord) computes also the discrete Legendre moments Md.
%   [M,Md,P1,P2]=cheb2moms(F,ord) returns the evaluation of orthogonal
%   polynomials on both axes.

n = ord;
type = 'Cheb2';
usesimpson = 1;	% use Simpson to compute integrals (otherwise, trapez. rule)

[m1 m2] = size(F);
if usesimpson && ~(2*round(m1/2)-m1), m1 = m1-1; end	% for Simpson
if usesimpson && ~(2*round(m2/2)-m2), m2 = m2-1; end	% for Simpson
F = double(F(1:m1,1:m2));
F = mat2gray(F);
x = linspace(-1,1,m1)';
y = linspace(-1,1,m2)';

[alfa beta] = opcoef(type,n);		% recursion coefficients
PP1 = opevmat(alfa,beta,x);		% values of polynomials on x
PP2 = opevmat(alfa,beta,y);		% values of polynomials on y
P1 = repmat(sqrt(1-x.^2),1,n+1) .* PP1;
P2 = repmat(sqrt(1-y.^2),1,n+1) .* PP2;

% continuous moments
M = opcmoms(F,P1,P2,usesimpson);
if nargout>1, Md = 4/m1/m2*(P1'*F*P2); end

