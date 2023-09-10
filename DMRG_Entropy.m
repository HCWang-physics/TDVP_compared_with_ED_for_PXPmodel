function [Onesite] = DMRG_Entropy(sWeight)
%UNTITLED8 此处提供此函数的摘要
%   此处提供详细说明
Site_number=length(sWeight)-1;
Onesite=zeros(1,Site_number);

for k = 1:Site_number
    Lambda=diag(sWeight{k}).^2;
    Onesite(1,k) = -sum(Lambda.*log(Lambda));
end
end