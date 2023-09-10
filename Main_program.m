
tic
%%%%% Simulation parameters for PXP model
d = 2; % local dimension
Nsites = 12; % number of lattice site`  s
Omega=1.33;
numval = d^Nsites; % number of eigenstates to compute
usePBCs=0;%Open boundary condition or peridoic boundary condition
%%%%% Define Hamiltonian (quantum PXP model)
H=sparse(zeros(d^Nsites));Sz_total=sparse(zeros(d^Nsites));
Sz_single={};Z2_total=sparse(zeros(d^Nsites));
% sX = [0,sqrt(2)/2,0;sqrt(2)/2,0,sqrt(2)/2;0,sqrt(2)/2,0];
% sY = [0,1j*sqrt(2)/2,0;-1j*sqrt(2)/2,0,1j*sqrt(2)/2;0,-1j*sqrt(2)/2,0];
% sZ = [1,0,0;0,0,0;0,0,-1];
% sI = eye(3);
sX = [0,1;1,0]; sY = [0,-1i;1i,0]; sZ = [1,0;0,-1]; sI = eye(2);
P=[0,0;0,1];
H_2_hopping=Omega*kron(P,kron(sX,P));
for i=1:Nsites-2
    H=H+kron(kron(speye(d^(i-1)),H_2_hopping),speye(d^(Nsites-i-2)));
    Sz_total=Sz_total+kron(kron(speye(d^(i-1)),sZ),speye(d^(Nsites-i)));
    Z2_total=Z2_total+kron(kron(speye(d^(i-1)),sX),speye(d^(Nsites-i)));
    Sz_single{i}=kron(kron(speye(d^(i-1)),sZ),speye(d^(Nsites-i)));
end
i=Nsites-1;
Sz_total=Sz_total+kron(kron(speye(d^(i-1)),sZ),speye(d^(Nsites-i)));
Z2_total=Z2_total+kron(kron(speye(d^(i-1)),sX),speye(d^(Nsites-i)));
Sz_single{i}=kron(kron(speye(d^(i-1)),sZ),speye(d^(Nsites-i)));
i=Nsites;
Sz_total=Sz_total+kron(kron(speye(d^(i-1)),sZ),speye(d^(Nsites-i)));
Z2_total=Z2_total+kron(kron(speye(d^(i-1)),sX),speye(d^(Nsites-i)));
Sz_single{i}=kron(kron(speye(d^(i-1)),sZ),speye(d^(Nsites-i)));
if usePBCs==1
    H=H+Omega*kron(kron(P,speye(d^(Nsites-3))),kron(P,sX));%2nd
    H=H+Omega*kron(kron(kron(sX,P),speye(d^(Nsites-3))),P);%2nd
end
%%%%% Do exact diag
[psi_eig,Energy0] = eigs(H,numval,'smallestreal');
Energy=sort(real(diag(Energy0)));
%%%Calculate entanglement
half_dim=d^(floor(Nsites/2));

for i=1:numval
    rho=reshape(psi_eig(:,i),floor(d^Nsites/half_dim),half_dim);
    rho=rho*rho';
    [Entropy] = eig(rho);
    SA(i)=-real(sum(log(Entropy).*Entropy));
    Sz_ob(i)=real(psi_eig(:,i)'*Sz_total*psi_eig(:,i));
    Z2_ob(i)=real(psi_eig(:,i)'*Z2_total*psi_eig(:,i));
end
Psi_ini=1;
up_psi=[1;0];down_psi=[0;1];
for i=1:Nsites
    Psi_ini=kron(Psi_ini,0.5*(1-(-1)^(i-1))*up_psi+0.5*(1-(-1)^(i-2))*down_psi);
end
Fidelity=abs(Psi_ini'*psi_eig).^2;
%%
%plot entropy of every eigenstate,Fidelity of |Z2> with all of
%eigenstates,Sz of every eigenstate, and Z2 of every eigenstate
%Final plot is the eigenvalue statistics
figure(1)
subplot(321)
plot(Energy,SA,'b.',markersize=8)
subplot(322)
plot(Energy,log(Fidelity),'b.',markersize=8)
subplot(323)
plot(Energy,Sz_ob,'r.',markersize=8)
subplot(324)
plot(Energy,Z2_ob,'r.',markersize=8)
subplot(313)
edges =[0:0.05:5];
deltaE=diff(Energy);
Ns = normalize(deltaE(abs(deltaE)>1e-4),"scale");
histogram(Ns,edges)
%% initial state is |Z2>
dT=0.04;Step=1000;
U_operator=expm(-1j*H*dT);
Psi_t=zeros(d^Nsites,Step);

Psi=Psi_ini;
Sz_t_1=zeros(Nsites,Step);
Entropy_t_1=zeros(1,Step);
for i=1:Step
    Psi=U_operator*Psi;
    Psi_t(:,i)=Psi;
    for k=1:Nsites
        Sz_t_1(k,i)=Psi'*Sz_single{k}*Psi;
    end
    rho=reshape(Psi,floor(d^Nsites/half_dim),half_dim);
    rho=rho*rho';
    [Entropy] = eig(rho);
    Entropy_t_1(i)=-real(sum(log(Entropy(Entropy>1e-10)).*Entropy(Entropy>1e-10)));
end
Fidelity_t_1=abs(Psi_ini'*Psi_t).^2;
%% initial state is |0>
dT=0.04;Step=1000;
U_operator=expm(-1j*H*dT);
Psi_t=zeros(d^Nsites,Step);
Psi_ini=1;
up_psi=[1;0];down_psi=[0;1];
for i=1:Nsites
    Psi_ini=kron(Psi_ini,down_psi);
end

Psi=Psi_ini;
Sz_t_2=zeros(Nsites,Step);
Entropy_t_2=zeros(1,Step);
for i=1:Step
    Psi=U_operator*Psi;
    Psi_t(:,i)=Psi;
    for k=1:Nsites
        Sz_t_2(k,i)=Psi'*Sz_single{k}*Psi;
    end
    rho=reshape(Psi,floor(d^Nsites/half_dim),half_dim);
    rho=rho*rho';
    [Entropy] = eig(rho);
    Entropy_t_2(i)=-real(sum(log(Entropy(Entropy>1e-10)).*Entropy(Entropy>1e-10)));
end
Fidelity_t_2=abs(Psi_ini'*Psi_t).^2;

%%
% mainDMRG_MPO
% ------------------------ 
%%%%% Example 1: Heisenberg model %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% Set simulation options
Band.chimin = 50; % maximum bond dimension
Band.chistep = 50; % maximum bond dimension
Band.chimax = 200; % maximum bond dimension
Nsites = 12; % number of lattice sites
chid=2;%local Hilbert space dimension
Omega=1.33;
%%%% Define Hamiltonian MPO (quantum XX model)
[A_initial,M_pre,M,ML,MR] = Get_MPO(chid,Band.chimin,Nsites,Omega);

%%%% Do DMRG sweeps
OPTS.numsweeps = 6; % number of DMRG sweeps
OPTS.display = 1; % level of output display
OPTS.updateon = 1; % update methond 1=Arnoldi 2=eigLanczos
OPTS.maxit = 2; % iterations of Lanczos method
OPTS.krydim = 4; % dimension of Krylov subspace
[A0,sWeight0,B0,Ekeep0] = doDMRG_MPO(A_initial,ML,M_pre,MR,OPTS,Band,chid);

TDVP.numsweeps = 1000; % number of time iteration
TDVP.midsweeps = 2; % number of time iteration
TDVP.tau = 0.02; % time step
TDVP.krydim=6; % dimension of Krylov subspace
[A,sWeight,B,Ob_Sz,Fidelity_t,Ob_Entropy,Cut_error] = do2TDVP_MPO(A0,ML,M,MR,TDVP,Band);
%%
%plot result and compared with ED
Time=1:Step;Time=Time*dT;
figure(2)
subplot(421)
plot(Time,Sz_t_1,linewidth=2)
subplot(423)
plot(Time,real(Ob_Sz),linewidth=2)
subplot(422)
plot(Time,Fidelity_t_1,linewidth=2)
hold on
plot(Time,Fidelity_t,linewidth=2)
legend('ED','TDVP')
xlabel('t')
ylabel('|<\Psi_{0}|\Psi(t)>|^2')
subplot(424)
plot(Time,Fidelity_t_1-Fidelity_t',linewidth=2)
ylabel('Errorr of |<\Psi_{0}|\Psi(t)>|^2')
subplot(425)
plot(Time,sum(Sz_t_1,1),linewidth=2)
hold on
plot(Time,sum(real(Ob_Sz),2),linewidth=2)
legend('ED','TDVP')
xlabel('t')
ylabel('S_{z}')
subplot(427)
plot(Time,sum(Sz_t_1,1)-sum(real(Ob_Sz),2)',linewidth=2)
ylabel('Error of S_{z}')
subplot(426)
plot(Time,Entropy_t_1,linewidth=2)
hold on
plot(Time,Ob_Entropy(:,floor(Nsites/2)+1),linewidth=2)
legend('ED','TDVP')
xlabel('t')
ylabel('S_{en}')
subplot(428)
plot(Time,Entropy_t_1-Ob_Entropy(:,floor(Nsites/2)+1)',linewidth=2)
ylabel('Errorr of S_{en}')
%%
toc