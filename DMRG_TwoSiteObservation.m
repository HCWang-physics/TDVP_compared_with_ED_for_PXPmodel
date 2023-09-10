function [Twosite_correlation] = DMRG_TwoSiteObservation(A,sWeight,Oper,Sele_site)
%UNTITLED8 此处提供此函数的摘要
%   此处提供详细说明
chid=size(Oper,1);
Two_operator = reshape(kron(Oper',Oper),[chid,chid,chid,chid]);
Site_number=length(A);
Twosite_correlation=zeros(1,Site_number-1);

Psi0=ncon({A{Sele_site},conj(A{Sele_site}),sWeight{Sele_site+1},conj(sWeight{Sele_site+1})},{[-1,-2,1],[-4,-3,3],[1,2],[3,2]});
for i=1:Sele_site-1
    Psi=Psi0;
    for j=Sele_site-1:-1:i
        if j==i
            Psi=ncon({A{j},conj(A{j}),Psi},{[1,-1,2],[1,-4,3],[2,-2,-3,3]});
        else
            Psi=ncon({A{j},conj(A{j}),Psi},{[-1,2,1],[-4,2,3],[1,-2,-3,3]});
        end
    end
    Twosite_correlation(i)=ncon({Psi,Two_operator},{[1,2,3,4],[3,4,2,1]});
end

Twosite_correlation(Sele_site)=ncon({A{Sele_site},Oper'*Oper,conj(A{Sele_site}), ...
    sWeight{Sele_site+1},conj(sWeight{Sele_site+1})},{[1,2,4],[3,2],[1,3,5],[4,6],[5,6]});

Psi0=ncon({A{Sele_site},conj(A{Sele_site})},{[1,-2,-1],[1,-3,-4]});
for i=Sele_site+1:Site_number
    Psi=Psi0;
    for j=Sele_site+1:i
        if j<i
            Psi=ncon({Psi,A{j},conj(A{j})},{[1,-2,-3,2],[1,3,-1],[2,3,-4]});
        else
            Psi=ncon({Psi,A{j},conj(A{j}),sWeight{j+1},conj(sWeight{j+1})},{[1,-2,-3,2],[1,-1,3],[2,-4,4],[3,5],[4,5]});
        end
    end
    Twosite_correlation(i)=ncon({Psi,Two_operator},{[1,2,3,4],[3,4,2,1]});
end

end