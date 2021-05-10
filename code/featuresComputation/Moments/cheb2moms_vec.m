function [v,i,j] = cheb2moms_vec(F,ord)
%CHEB2MOMS_VEC vector of second kind Chebyshev moments of an image
%   v=cheb2moms_vec(F,ord) computes the vector v of the second kind Chebyshev
%   moments of the image F, up to order ord.

M = cheb2moms(F,ord);
idx = fliplr(triu(ones(size(M))));
v = M(logical(idx))';

if nargout > 1 
    idx2 = find(idx);
    [i,j] = ind2sub(size(M),idx2);
    %v = M(sub2ind(size(M),j,i));
    i = i-1;
    j = j-1;
end