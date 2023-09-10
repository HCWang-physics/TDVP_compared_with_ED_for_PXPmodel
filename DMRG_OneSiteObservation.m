function [Onesite] = DMRG_OneSiteObservation(A,sWeight,Oper)
%UNTITLED8 此处提供此函数的摘要
%   此处提供详细说明
Site_number=length(A);
Onesite=zeros(1,Site_number);

for k = 1:Site_number
    rhotwo = ncon({A{k},conj(A{k}),sWeight{k+1},sWeight{k+1}},...
        {[1,-2,2],[1,-1,3],[2,4],[3,4]});
    Onesite(1,k) = ncon({Oper,rhotwo},{[1,2],[1,2]});
end
end