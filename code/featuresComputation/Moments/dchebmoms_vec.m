function [v,i,j] = dchebmoms_vec(F,ord)
%DCHEBMOMSS_VEC vector of discrete Chebyshev moments of an image
%   v=dchebmoms_vec(F,ord) computes the vector v of the discrete Chebyshev
%   moments of the image F, up to order ord.

M = dchebmoms(F,ord);
idx = fliplr(triu(ones(size(M))));
v = M(logical(idx))';


if nargout > 1 
    idx2 = find(idx);
    [i,j] = ind2sub(size(M),idx2);
    %v = M(sub2ind(size(M),j,i));
    i = i-1;
    j = j-1;
end