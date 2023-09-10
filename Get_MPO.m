function [A_initial,M_pre,M,ML,MR] = Get_MPO(chid,chi,Nsites,Omega)
%%%% Define Hamiltonian MPO (quantum XX model)
Sx = [0, 1; 1, 0];
Sy = [0, -1i; 1i, 0];
Sz = [1, 0; 0, -1];
sI = eye(2);
P=[0, 0; 0, 1];

M_mid = zeros(4,4,2,2);
M_mid(1,1,:,:) = sI;M_mid(1,2,:,:) = Omega*P;M_mid(4,4,:,:) = sI;
M_mid(2,3,:,:) = Sx;M_mid(3,4,:,:) = P;
Mboundary = zeros(2,2,2,2);
Mboundary(1,1,:,:) = sI;Mboundary(1,2,:,:) = Omega*P;
Mboundary(2,2,:,:) = sI;
ML1 = zeros(2,3,2,2);
ML1(1,1,:,:) = sI;ML1(1,2,:,:) = Omega*P;
ML1(2,3,:,:) = Sx;
ML2 = zeros(3,4,2,2);
ML2(1,1,:,:) = sI;ML2(1,2,:,:) = Omega*P;
ML2(2,3,:,:) = Sx;ML2(3,4,:,:) = P;

MR1 = zeros(3,2,2,2);
MR1(1,1,:,:) = Sx;MR1(2,2,:,:) = P;
MR1(3,2,:,:) = sI;
MR2 = zeros(4,3,2,2);
MR2(1,1,:,:) = P;MR2(2,2,:,:) = Sx;
MR2(3,3,:,:) = P;MR2(4,3,:,:) = sI;
ML = reshape([1;0;],[2,1,1]); %left MPO boundary
MR = reshape([0;1],[2,1,1]); %right MPO boundary

%%%% Initialize MPS tensors
A_initial = {};
A_initial{1} = rand(1,chid,min(chi,chid));
for k = 2:Nsites
    A_initial{k} = rand(size(A_initial{k-1},3),chid,min(min(chi,size(A_initial{k-1},3)*chid),chid^(Nsites-k)));
end
M = {};M{1}=Mboundary;M{Nsites}=Mboundary;
M{2}=ML1;M{3}=ML2;
M{Nsites-1}=MR1;M{Nsites-2}=MR2;
for k = 4:Nsites-3
    M{k}=M_mid;
end

hz=1;
M_pre={};
for k = 1:Nsites
    Mpre = zeros(2,2,2,2);
    Mpre(1,1,:,:) = sI;Mpre(1,2,:,:) = (-1)^(k-1)*hz*Sz;%(-1)^(k-1)*hz*Sz;
    Mpre(2,2,:,:) = sI;
    M_pre{k}=Mpre;
end
end