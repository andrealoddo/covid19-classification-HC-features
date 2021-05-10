function [v, i, j] = legmoms_vec2(F,ord)
%LEGMOMS_VEC vector of Legendre moments of an image
%   v=legmoms_vec(F,ord) computes the vector v of the continuous Legendre
%   moments of the image F, up to order ord.

M = legmoms(F,ord);
idx = fliplr(triu(ones(size(M))));
v = M(logical(idx))';

idx2 = find(idx);
[i,j] = ind2sub(size(M),idx2);
i = i-1;
j = j-1;
