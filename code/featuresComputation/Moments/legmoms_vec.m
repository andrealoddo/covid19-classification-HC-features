function [v,i,j] = legmoms_vec(F,ord,usesimpson)
%LEGMOMS_VEC vector of Legendre moments of an image
%   v=legmoms_vec(F,ord) computes the vector v of the continuous Legendre
%   moments of the image F, up to order ord.

if nargin == 2
    usesimpson = 1;	% use Simpson to compute integrals (otherwise, trapez. rule)
end

M = legmoms(F,ord,usesimpson);
idx = fliplr(triu(ones(size(M))));
v = M(logical(idx))';

if nargout > 1 
    idx2 = find(idx);
    [i,j] = ind2sub(size(M),idx2);
    %v = M(sub2ind(size(M),j,i));
    i = i-1;
    j = j-1;
end