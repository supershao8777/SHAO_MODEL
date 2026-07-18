function [R, Ps, Q, A, S, obj_history] = bcd_mvrl13(X, m, c, lambda, gamma, eta, opts)
% BCD_MVRL13  Multi-View RL (Stiefel Ps + Sylvester Q + decoupling η).
%
%   [R, Ps, Q, A, S, obj] = bcd_mvrl13(X, m, c, lambda, gamma, eta, opts)
%
%   Objective:
%     J = Σ_v ||X^{(v)} - R^{(v)}(Ps+Q^{(v)})A^{(v)}||_F^2
%       + λ Σ_v ||(Ps+Q^{(v)})A^{(v)} - S||_F^2
%       + γ Σ_v ||A^{(v)}||_F^2
%       + η Σ_v ||Ps^T Q^{(v)}||_F^2                       ← decoupling
%
%   Constraints:
%     R{v}: d_v×m,  R^T R = I_m              (Procrustes)
%     Ps  : m×c,    Ps^T Ps = I_c             (Stiefel GD + SVD)
%     Q{v}: m×c,    unconstrained             (Sylvester)
%     A{v}: c×n,    col-simplex               (closed-form + simplex)
%     S   : m×n,    S S^T = I_m               (row-orthogonal, Procrustes)
%
%   Update order: R → A → S → Ps → Q
%   Data: X{v} is d_v × n

%% Options
if nargin < 7, opts = struct(); end
max_iter  = get_opt(opts, 'max_iter',  50);
pgd_steps = get_opt(opts, 'pgd_steps', 10);
eta_Ps    = get_opt(opts, 'eta_Ps',    1e-3);
tol       = get_opt(opts, 'tol',       1e-4);
verbose   = get_opt(opts, 'verbose',   true);

V = length(X); [~, n] = size(X{1}); eps_reg = 1e-6;

if verbose
    fprintf('\n========== BCD-MVRL13 ==========\n');
    fprintf('n=%d, m=%d, c=%d, V=%d, λ=%.4f, γ=%.4f, η=%.4f\n', n, m, c, V, lambda, gamma, eta);
    fprintf('Ps: Stiefel GD+SVD, Q: Sylvester, S: S S^T=I_m (Orthogonal Procrustes)\n');
    fprintf('Update: R → A → S → Ps → Q\n');
end
for v = 1:V, assert(size(X{v},2)==n, 'Sample mismatch.'); end

%% Init
rng(42, 'twister');
R = cell(1,V);
for v=1:V
    dv=size(X{v},1);
    if dv>=m,[R{v},~]=qr(randn(dv,m),0);else R{v}=randn(dv,m)/sqrt(dv);end
end
[Ps,~]=qr(randn(m,c),0);
Q=cell(1,V);for v=1:V,Q{v}=randn(m,c)*0.01;end
A=cell(1,V);for v=1:V,A{v}=proj_simplex(rand(c,n));end
[S_tmp,~]=qr(randn(n,m),0);S=S_tmp';  % S S^T = I_m (row-orthogonal)

obj_old=Inf;obj_history=zeros(max_iter,1);
if verbose,fprintf('\n  Iter   Obj            RelChg\n  ----  --------------  ------\n');end

%% BCD
for iter=1:max_iter
    %% Step 1: R (Orthogonal Procrustes)
    for v=1:V
        Yv=(Ps+Q{v})*A{v};
        [U_R,~,V_R]=svd(X{v}*Yv','econ');
        R{v}=U_R*V_R';
    end

    %% Step 2: A (closed-form + simplex)
    %   A = ((1+λ) Z^T Z + γI)^{-1} (Z^T R^T X + λ Z^T S)
    for v=1:V
        Zv=Ps+Q{v};
        A{v}=((1+lambda)*(Zv'*Zv)+gamma*eye(c)+eps_reg*eye(c))\...
             (Zv'*R{v}'*X{v}+lambda*Zv'*S);
        A{v}=proj_simplex(A{v});
    end

    %% Step 3: S (Orthogonal Procrustes: S S^T = I_m)
    %   F = (1/V) Σ (Ps+Q)A [m×n]
    %   SVD(F) → S = U V^T
    F_S=zeros(m,n);
    for v=1:V,F_S=F_S+(Ps+Q{v})*A{v};end
    F_S=F_S/V;
    [U_S,~,V_S]=svd(F_S,'econ');S=U_S*V_S';           % S S^T = I_m

    %% Step 4: Ps (Stiefel QP: min Tr(Ps^T M_left Ps + Ps M_right Ps^T) - 2Tr(Ps^T N))
    %   M_left = η Σ Q Q^T [m×m],  M_right = (1+λ) Σ A A^T [c×c]
    %   N = Σ(R^T X A^T + λ S A^T - (1+λ) Q A A^T) [m×c]
    %   ∇Ps = 2 M_left Ps + 2 Ps M_right - 2 N
    M_left=zeros(m,m);M_right=zeros(c,c);N_Ps=zeros(m,c);
    for v=1:V
        M_left=M_left+eta*(Q{v}*Q{v}');
        M_right=M_right+(1+lambda)*(A{v}*A{v}');
        N_Ps=N_Ps+R{v}'*X{v}*A{v}'+lambda*S*A{v}'-(1+lambda)*Q{v}*A{v}*A{v}';
    end
    for pgd=1:pgd_steps
        grad_Ps=2*M_left*Ps+2*Ps*M_right-2*N_Ps;
        Ps=Ps-eta_Ps*grad_Ps;
        [U_P,~,V_P]=svd(Ps,'econ');Ps=U_P*V_P';
    end

    %% Step 5: Q (Sylvester: (η Ps Ps^T)·Q + Q·((1+λ) A A^T) = C_syl)
    A_syl=eta*(Ps*Ps');
    for v=1:V
        B_syl=(1+lambda)*(A{v}*A{v}');
        C_syl=R{v}'*X{v}*A{v}'+lambda*S*A{v}'-(1+lambda)*Ps*A{v}*A{v}';
        K=kron(eye(c),A_syl)+kron(B_syl',eye(m));
        Q{v}=reshape(K\C_syl(:),[m,c]);
    end

    %% Objective
    obj=0;
    for v=1:V
        Mv=Ps+Q{v};
        obj=obj+norm(X{v}-R{v}*Mv*A{v},'fro')^2;
        obj=obj+lambda*norm(Mv*A{v}-S,'fro')^2;
        obj=obj+gamma*norm(A{v},'fro')^2;
        obj=obj+eta*norm(Ps'*Q{v},'fro')^2;
    end
    obj_history(iter)=obj;
    rel_chg=abs(obj_old-obj)/max(1,abs(obj_old));
    if verbose&&(mod(iter,5)==1||iter==1)
        fprintf('  %4d  %14.6e  %6.2e\n',iter,obj,rel_chg);
    end
    if rel_chg<tol,if verbose,fprintf('  Converged iter %d.\n',iter);end
        obj_history=obj_history(1:iter);break;end
    obj_old=obj;
end
if verbose&&iter>=max_iter,fprintf('  Max iters.\n');end
if verbose,fprintf('=== Summary: Iter=%d Obj=%.4e\n',iter,obj);end
end

%% Helpers
function X=proj_simplex(Z)
[M,N]=size(Z);X=zeros(M,N);
for j=1:N
    z=Z(:,j);u=sort(z,'descend');cs=cumsum(u);
    rho=find((1:M)'.*u>(cs-1),1,'last');
    if isempty(rho),rho=1;end
    theta=(cs(rho)-1)/rho;X(:,j)=max(z-theta,0);
end
end

function v=get_opt(o,f,d)
if isfield(o,f),v=o.(f);else v=d;end
end
