function [R, Ps, Q, A, obj_history] = bcd_mvrl17(X, m, c, beta, opts)
% BCD_MVRL17  Multi-View RL (ridge Q + Q≥0 + Ps Stiefel, no S).
%
%   [R, Ps, Q, A, obj] = bcd_mvrl17(X, m, c, beta, opts)
%
%   Objective:
%     J = Σ_v ||X^{(v)} - R^{(v)}(Ps+Q^{(v)})A^{(v)}||_F^2
%       + β Σ_v ||Q^{(v)}||_F^2                       ← ridge on Q
%
%   Constraints:
%     R{v}: d_v×m,  R^T R = I_m              (Procrustes)
%     Ps  : m×c,    Ps^T Ps = I_c             (Procrustes via R^T left-multiply)
%     Q{v}: m×c,    Q ≥ 0                     (PGD + ReLU)
%     A{v}: c×n,    col-simplex               (PGD + simplex)
%
%   NOTE: NO S. β||Q||² ridge on Q (no Ps^T Q, no A ridge).
%   Update order: R → A → Q → Ps
%   Data: X{v} is d_v × n

%% Options
if nargin < 5, opts = struct(); end
max_iter  = get_opt(opts, 'max_iter',  50);
pgd_steps = get_opt(opts, 'pgd_steps', 5);
tol       = get_opt(opts, 'tol',       1e-4);
verbose   = get_opt(opts, 'verbose',   true);

V = length(X); [~, n] = size(X{1}); eps_reg = 1e-6;

if verbose
    fprintf('\n========== BCD-MVRL17 ==========\n');
    fprintf('n=%d, m=%d, c=%d, V=%d, β=%.4f\n', n, m, c, V, beta);
    fprintf('NO S. β||Q||² ridge on Q\n');
    fprintf('R: Procrustes, A/Q: PGD, Ps: OP via R^T left-multiply\n');
    fprintf('Update: R → A → Q → Ps\n');
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

obj_old=Inf;obj_history=zeros(max_iter,1);
if verbose,fprintf('\n  Iter   Obj            RelChg\n  ----  --------------  ------\n');end

%% BCD
for iter=1:max_iter
    %% Step 1: R (Orthogonal Procrustes)
    for v=1:V
        Mv=(Ps+Q{v})*A{v};
        [U_R,~,V_R]=svd(X{v}*Mv','econ');
        R{v}=U_R*V_R';
    end

    %% Step 2: A (PGD + simplex)
    %   B = R(Ps+Q),  ∇A = 2 B^T(B A - X)
    for v=1:V
        Bv=R{v}*(Ps+Q{v});
        L_A=2*norm(Bv'*Bv,2);eta_A=1/max(L_A,eps_reg);
        for pgd=1:pgd_steps
            grad_A=2*Bv'*(Bv*A{v}-X{v});
            A{v}=A{v}-eta_A*grad_A;
            A{v}=proj_simplex(A{v});
        end
    end

    %% Step 3: Q (PGD + ReLU, with ridge)
    %   ∇Q = 2 R^T(R(Ps+Q)A - X)A^T + 2β Q
    for v=1:V
        L_Q=2*(norm(R{v}'*R{v},2)*norm(A{v}*A{v}',2)+beta);
        eta_Q=1/max(L_Q,eps_reg);
        for pgd=1:pgd_steps
            Mv=Ps+Q{v};
            grad_Q=2*R{v}'*(R{v}*Mv*A{v}-X{v})*A{v}'+2*beta*Q{v};
            Q{v}=max(Q{v}-eta_Q*grad_Q,0);
        end
    end

    %% Step 4: Ps (Orthogonal Procrustes via R^T left-multiply)
    %   ||R^T X - (Ps+Q)A||² = ||Ps A - (R^T X - Q A)||²
    %   H = Σ_v (R_v^T X_v - Q_v A_v) A_v^T  [m×c]
    %   SVD(H) → Ps = U V^T
    H_Ps=zeros(m,c);
    for v=1:V
        H_Ps=H_Ps+(R{v}'*X{v}-Q{v}*A{v})*A{v}';
    end
    [U_P,~,V_P]=svd(H_Ps,'econ');Ps=U_P*V_P';

    %% Objective
    obj=0;
    for v=1:V
        obj=obj+norm(X{v}-R{v}*(Ps+Q{v})*A{v},'fro')^2;
        obj=obj+beta*norm(Q{v},'fro')^2;
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
