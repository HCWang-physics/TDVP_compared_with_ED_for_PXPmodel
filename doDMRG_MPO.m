function [A,sWeight,B,Ekeep] = doDMRG_MPO(A,ML,M,MR,OPTS,Band,chid)
Sx = [0, 1; 1, 0];
Sy = [0, -1i; 1i, 0];
Sz = [1, 0; 0, -1];
sI = eye(2);
P=[0, 0; 0, 1];


%%%%% left-to-right 'warmup', put MPS in right orthogonal form
Nsites = length(A);
L{1} = ML; R{Nsites} = MR;
for p = 1:Nsites - 1
    chil = size(A{p},1);chid = size(A{p},2); chir = size(A{p},3);
    [qtemp,rtemp] = qr(reshape(A{p},[chil*chid,chir]),0);
    A{p} = reshape(qtemp,[chil,chid,chir]);
    A{p+1} = ncon({rtemp,A{p+1}},{[-1,1],[1,-2,-3]})/norm(rtemp(:));
    L{p+1} = ncon({L{p},M{p},A{p},conj(A{p})},{[2,1,4],[2,-1,3,5],[4,5,-3],[1,3,-2]});
end
chil = size(A{Nsites},1); chir = size(A{Nsites},3);
[qtemp,stemp] = qr(reshape(A{Nsites},[chil*chid,chir]),0);
A{Nsites} = reshape(qtemp,[chil,chid,chir]);
sWeight{Nsites+1} = stemp./sqrt(trace(stemp*stemp'));
chimax=size(A{floor(Nsites/2)},1);
Ekeep = [];
for k = 1:OPTS.numsweeps
    error=0;
    %%%%%% Optimization sweep: right-to-left 
    for p = Nsites-1:-1:1
        %%%%% two-site update
        chil = size(A{p},1); chir = size(A{p+1},3); chid1 = size(A{p},2); chid2 = size(A{p+1},2);
        psiGround = reshape(ncon({A{p},A{p+1},sWeight{p+2}},{[-1,-2,1],[1,-3,2],[2,-4]}),[chil*chid1*chid2*chir,1]);
        if OPTS.updateon==1
            [psiGround,Ekeep(end+1)] = eig_Arnoldi(psiGround,OPTS,@doApplyMPO,{L{p},M{p},M{p+1},R{p+1}});
        else
            [psiGround,Ekeep(end+1)] = eigLanczos(psiGround,OPTS,@doApplyMPO,{L{p},M{p},M{p+1},R{p+1}});
        end
        [utemp,stemp,vtemp] = svd(reshape(psiGround,[chil*chid1,chid2*chir]),'econ');
        chi_band=max(size(A{p},3),chimax);
        chitemp = min(min(size(stemp)),chi_band);
        A{p} = reshape(utemp(:,1:chitemp),[chil,chid1,chitemp]);
        Mode0=sqrt(sum(diag(stemp(1:chitemp,1:chitemp)).^2));Mode_all=sqrt(sum(diag(stemp).^2));
        error=error+abs(Mode_all-Mode0)/Mode_all;
        sWeight{p+1} = stemp(1:chitemp,1:chitemp)./Mode0;
        B{p+1} = reshape(vtemp(:,1:chitemp)',[chitemp,chid2,chir]);  
        %%%%% new block Hamiltonian MPO
        R{p} = ncon({M{p+1},R{p+1},B{p+1},conj(B{p+1})},{[-1,2,3,5],[2,1,4],[-3,5,4],[-2,3,1]});
       %%%%% display energy
        if OPTS.display == 2
            fprintf('Sweep: %2.1d of %2.1d, Loc: %2.1d, Energy: %12.12d\n',k,OPTS.numsweeps,p,Ekeep(end));
        end 
    end
    %%%%%% left boundary tensor
    chil = size(A{1},1); chid = size(A{1},2); chir = size(A{1},3);
    [utemp,stemp,vtemp] = svd(reshape(ncon({A{1},sWeight{2}},{[-1,-2,1],[1,-3]}),[chil,chid*chir]),'econ');
    B{1} = reshape(vtemp',[chil,chid,chir]);
    sWeight{1} = utemp*stemp./sqrt(trace(stemp.^2));
    %%%%%% Optimization sweep: left-to-right
    for p = 1:Nsites-1
        %%%%% two-site update
        chil = size(B{p},1); chid1 = size(B{p},2); chid2 = size(B{p+1},2); chir = size(B{p+1},3);
        psiGround = reshape(ncon({sWeight{p},B{p},B{p+1}},{[-1,1],[1,-2,2],[2,-3,-4]}),[chil*chid1*chid2*chir,1]);
        if OPTS.updateon==1
            [psiGround,Ekeep(end+1)] = eig_Arnoldi(psiGround,OPTS,@doApplyMPO,{L{p},M{p},M{p+1},R{p+1}});
        else
            [psiGround,Ekeep(end+1)] = eigLanczos(psiGround,OPTS,@doApplyMPO,{L{p},M{p},M{p+1},R{p+1}});
        end
        [utemp,stemp,vtemp] = svd(reshape(psiGround,[chil*chid1,chid2*chir]),'econ');
        chi_band=max(size(A{p},3),chimax);
        chitemp = min(min(size(stemp)),chi_band);
        A{p} = reshape(utemp(:,1:chitemp),[chil,chid1,chitemp]);
        Mode0=sqrt(sum(diag(stemp(1:chitemp,1:chitemp)).^2));Mode_all=sqrt(sum(diag(stemp).^2));
        error=error+abs(Mode_all-Mode0)/Mode_all;
        sWeight{p+1} = stemp(1:chitemp,1:chitemp)./Mode0;
        B{p+1} = reshape(vtemp(:,1:chitemp)',[chitemp,chid2,chir]);
        %%%%% new block Hamiltonian
        L{p+1} = ncon({L{p},M{p},A{p},conj(A{p})},{[2,1,4],[2,-1,3,5],[4,5,-3],[1,3,-2]});
        %%%%% display energy
        if OPTS.display == 2
            fprintf('Sweep: %2.1d of %2.1d, Loc: %2.1d, Energy: %12.12d\n',k,OPTS.numsweeps,p,Ekeep(end));
        end
    end
    %%%%%% right boundary tensor
    chil = size(B{Nsites},1); chid = size(B{Nsites},2); chir = size(B{Nsites},3);
    [utemp,stemp,vtemp] = svd(reshape(ncon({B{Nsites},sWeight{Nsites}},{[1,-2,-3],[-1,1]}),[chil*chid,chir,1]),'econ');    
    A{Nsites} = reshape(utemp,[chil,chid,chir]);
    sWeight{Nsites+1} = (stemp./sqrt(sum(diag(stemp).^2)))*vtemp';
    if error>1e-8
        chimax=min(chimax+Band.chistep,Band.chimax);
    elseif error<1e-8
        chimax=max(chimax-Band.chistep,Band.chimin);
    end
    %%%%%Obeserve
    [Onesite] = DMRG_OneSiteObservation(A,sWeight,Sz);
    if OPTS.display == 1
        fprintf('Sweep: %2.1d of %2.1d,CutEr: %4.4d, ReE: %4.4d, ImE: %4.4d, ReSz: %4.4d, MaxSz: %4.4d, ImSz: %4.4d, Band: %4.4d\n',k,OPTS.numsweeps,...
            error,real(Ekeep(end)),imag(Ekeep(end)),real(sum(Onesite)),max(real(Onesite)),imag(sum(Onesite)),chimax);
    end
end
% A{Nsites} = ncon({A{Nsites},sWeight{Nsites+1}},{[-1,-2,1],[1,-3]});
% sWeight{Nsites+1} = eye(size(A{Nsites},3));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function psi = doApplyMPO(psi,L,M1,M2,R)
% applies the superblock MPO to the state

psi = reshape(ncon({reshape(psi,[size(L,3),size(M1,4),size(M2,4),size(R,3)]),L,M1,M2,R},...
    {[1,3,5,7],[2,-1,1],[2,4,-2,3],[4,6,-3,5],[6,-4,7]}),[size(L,3)*size(M1,4)*size(M2,4)*size(R,3),1]);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [psivec,dval] = eigLanczos(psivec,OPTS,linFunct,functArgs)
% function for computing the smallest algebraic eigenvalue and eigenvector
% of the linear function 'linFunct' using a Lanczos method. Maximum
% iterations are specified by 'OPTS.maxit' and the dimension of Krylov
% space is specified by 'OPTS.krydim'. Input 'functArgs' is an array of
% optional arguments passed to 'linFunct'.

if norm(psivec) == 0
    psivec = rand(length(psivec),1);
end
psi = zeros(numel(psivec),OPTS.krydim+1);
A = zeros(OPTS.krydim,OPTS.krydim);
for k = 1:OPTS.maxit
    
    psi(:,1) = psivec(:)/norm(psivec);
    for p = 2:OPTS.krydim+1
        psi(:,p) = linFunct(psi(:,p-1),functArgs{(1:length(functArgs))});
        for g = 1:1:p-1
            A(p-1,g) = dot(psi(:,p),psi(:,g));
            A(g,p-1) = conj(A(p-1,g));
        end
        for g = 1:1:p-1
            psi(:,p) = psi(:,p) - dot(psi(:,g),psi(:,p))*psi(:,g);
            psi(:,p) = psi(:,p)/max(norm(psi(:,p)),1e-16);
        end
    end
    
    [utemp,dtemp] = eig(A);
    Evalue=real(diag(dtemp));
    xloc = find(Evalue == min(Evalue));
    psivec = psi(:,1:OPTS.krydim)*utemp(:,xloc(1));
end
psivec = psivec/norm(psivec);
dval = dtemp(xloc(1),xloc(1));

function [psivec,dval] = eig_Arnoldi(psivec,OPTS,linFunct,functArgs)
% function for computing the smallest algebraic eigenvalue and eigenvector
% of the linear function 'linFunct' using a Lanczos method. Maximum
% iterations are specified by 'OPTS.maxit' and the dimension of Krylov
% space is specified by 'OPTS.krydim'. Input 'functArgs' is an array of
% 114514
% optional arguments passed to 'linFunct'.

if norm(psivec) == 0
    psivec = rand(length(psivec),1);
end
psi = zeros(numel(psivec),OPTS.krydim+1);
A = zeros(OPTS.krydim+1,OPTS.krydim);
for k = 1:OPTS.maxit
    psi(:,1) = psivec(:)/norm(psivec);
    for p = 2:OPTS.krydim+1
        psi0 = linFunct(psi(:,p-1),functArgs{(1:length(functArgs))});
        psi(:,p) = psi0;
        for g = 1:1:p-1
            A(g,p-1)=dot(psi(:,g),psi0);
        end
        for g = 1:1:p-1
            h=dot(psi(:,g),psi(:,p));
            psi(:,p) = psi(:,p) - h*psi(:,g);
            psi(:,p) = psi(:,p)/max(norm(psi(:,p)),1e-16);
        end
        A(p,p-1)=dot(psi(:,p),psi0);
    end

    [Rutemp,dtemp,~] = eig(A(1:OPTS.krydim,1:OPTS.krydim));
    Evalue=real(diag(dtemp));
    xloc = find(Evalue == min(Evalue));
    psivec = psi(:,1:OPTS.krydim)*Rutemp(:,xloc(1));
end
psivec = psivec/norm(psivec);
dval = dtemp(xloc(1),xloc(1));