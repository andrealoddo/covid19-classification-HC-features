function [a1, b2] = shiftMatrices(a, b, offset)

[y,x]=size(a);

if ((y <= offset(1)) || (x <= offset(2)))
    error('The offsets should be minor than matrix sizes');
end

a1 = a;
b2 = b;

if offset(1) > 0
    b2 = b2(1+offset(1):end,:);
    a1 = a1(1:end-offset(1),:); 
elseif offset(1) < 0
    a1 = a1(1-offset(1):end,:);
    b2 = b2(1:end+offset(1),:);
end

if offset(2) > 0
    b2 = b2(:,1+offset(2):end);
    a1 = a1(:,1:end-offset(2));
elseif offset(2) < 0
    a1 = a1(:,1-offset(2):end);
    b2 = b2(:,1:end+offset(2));
end