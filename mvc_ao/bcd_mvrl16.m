function [R, Ps, Q, A, S, obj_history] = bcd_mvrl16(X, m, c, beta, gamma, opts)
% BCD_MVRL16  Multi-View RL (S orthogonal + Q≥0 + Ps Stiefel).
%
%   [R, Ps, Q, A, S, obj] = bcd_mvrl16(X, m, c, beta, gamma, opts)
%
%   Objective:
%     J = Σ_v ||X^{(v)} - R^{(v)}(Ps+Q^{(v)})A^{(v)}||_F^2
%       + β Σ_v ||Ps A^{(v)} - S||_F^2           ← Ps-only consensus
%       + γ Σ_v ||Ps^T Q^{(v)}||_F^2             ← decoupling
%
%   Constraints:
%     R{v}: d_v×m,  R^T R = I_m              (Procrustes)
%     Ps  : m×c,    Ps^T Ps = I_c             (Stiefel tangent + SVD)
%     Q{v}: m×c,    Q ≥ 0                     (PGD + ReLU)
%     A{v}: c×n,    col-simplex               (PGD + simplex)
%     S   : m×n,    S S^T = I_m               (row-orthogonal, Procrustes)
%
%   Update order: R → A → Q → S → Ps
%   Data: X{v} is d_v × n

%% Options
if nargin < 6, opts = struct(); end
max_iter  = get_opt(opts, 'max_iter',  50);
pgd_steps = get_opt(opts, 'pgd_steps', 5);
eta_Ps    = get_opt(opts, 'eta_Ps',    1e-3);
tol       = get_opt(opts, 'tol',       1e-4);
verbose   = get_opt(opts, 'verbose',   true);

V = length(X); [~, n] = size(X{1}); eps_reg = 1e-6;

if verbose
    fprintf('\n========== BCD-MVRL16 ==========\n');
    fprintf('n=%d, m=%d, c=%d, V=%d, β=%.4f, γ=%.4f\n', n, m, c, V, beta, gamma);
    fprintf('S: S S^T=I_m (Procrustes), Q≥0, Ps: Stiefel tangent\n');
    fprintf('Update: R → A → Q → S → Ps\n');
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
Q=cell(1,V);for v=1:V,Q{v}=max(rand(m,c)*0.1,0);end
A=cell(1,V);for v=1:V,A{v}=proj_simplex(rand(c,n));end
[S_tmp,~]=qr(randn(n,m),0);S=S_tmp';  % S S^T = I_m

obj_old=Inf;obj_history=zeros(max_iter,1);
if verbose,fprintf('\n  Iter   Obj            RelChg\n  ----  --------------  ------\n');end

%% BCD
for iter=1:max_iter
    %% Step 1: R (Orthogonal Procrustes)
    for v=1:V
        Mv=(Ps+Q{v})*A{v};                           % m×n
        [U_R,~,V_R]=svd(X{v}*Mv','econ');            % d_v×m
        R{v}=U_R*V_R';
    end

    %% Step 2: A (PGD + simplex)
    %   ∇A = 2 Z^T R^T(R Z A - X) + 2β Ps^T(Ps A - S)
    for v=1:V
        Zv=Ps+Q{v};
        Hv=R{v}*Zv;
        L_A=2*(norm(Hv'*Hv,2)+beta*norm(Ps'*Ps,2));eta_A_l=1/max(L_A,eps_reg);
        for pgd=1:pgd_steps
            grad_A=2*Zv'*R{v}'*(R{v}*Zv*A{v}-X{v})+2*beta*Ps'*(Ps*A{v}-S);
            A{v}=A{v}-eta_A_l*grad_A;
            A{v}=proj_simplex(A{v});
        end
    end

    %% Step 3: Q (PGD + ReLU)
    %   ∇Q = 2 R^T(R(Ps+Q)A - X)A^T + 2γ Ps Ps^T Q
    for v=1:V
        L_Q=2*(norm(R{v}'*R{v},2)*norm(A{v}*A{v}',2)+gamma*norm(Ps*Ps',2));
        eta_Q_l=1/max(L_Q,eps_reg);
        for pgd=1:pgd_steps
            Mv=Ps+Q{v};
            grad_Q=2*R{v}'*(R{v}*Mv*A{v}-X{v})*A{v}'+2*gamma*Ps*Ps'*Q{v};
            Q{v}=max(Q{v}-eta_Q_l*grad_Q,0);
        end
    end

    %% Step 4: S (Orthogonal Procrustes: S S^T = I_m)
    %   G = (1/V) Σ Ps A_v [m×n]
    %   SVD(G) → S = U V^T
    G_S=zeros(m,n);
    for v=1:V,G_S=G_S+Ps*A{v};end
    G_S=G_S/V;
    [U_S,~,V_S]=svd(G_S,'econ');S=U_S*V_S';

    %% Step 5: Ps (Stiefel tangent + SVD retraction)
    %   ∇Ps = Σ [2 R^T(R(Ps+Q)A-X)A^T + 2β(Ps A-S)A^T + 2γ Q Q^T Ps]
    %   Tangent projection: G_proj = ∇Ps - Ps·(∇Ps^T·Ps)
    %   Retraction: SVD(Ps - η·G_proj) → U V^T
    for pgd=1:pgd_steps
        grad_Ps=zeros(m,c);
        for v=1:V
            Mv=Ps+Q{v};
            grad_Ps=grad_Ps+2*R{v}'*(R{v}*Mv*A{v}-X{v})*A{v}';
            grad_Ps=grad_Ps+2*beta*(Ps*A{v}-S)*A{v}';
            grad_Ps=grad_Ps+2*gamma*Q{v}*Q{v}'*Ps;
        end
        % Tangent space projection
        G_proj=grad_Ps-Ps*(grad_Ps'*Ps);
        Ps=Ps-eta_Ps*G_proj;
        [U_P,~,V_P]=svd(Ps,'econ');Ps=U_P*V_P';
    end

    %% Objective
    obj=0;
    for v=1:V
        obj=obj+norm(X{v}-R{v}*(Ps+Q{v})*A{v},'fro')^2;
        obj=obj+beta*norm(Ps*A{v}-S,'fro')^2;
        obj=obj+gamma*norm(Ps'*Q{v},'fro')^2;
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
