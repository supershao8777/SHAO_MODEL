function [R, Ps, Q, Z, S, obj_history] = bcd_mvrl12(X, m, c, lambda, opts)
% BCD_MVRL12  Multi-View RL (Q≥0, closed-form Z, OP Ps + simplex S).
%
%   [R, Ps, Q, Z, S, obj] = bcd_mvrl12(X, m, c, lambda, opts)
%
%   Objective:
%     J = Σ_v ||X^{(v)} - R^{(v)}(Ps+Q^{(v)})Z^{(v)}||_F^2
%       + λ Σ_v ||Ps Z^{(v)} - S||_F^2               ← Ps-only consensus
%
%   Constraints:
%     R{v}: d_v×m,  R^T R = I_m              (Orthogonal Procrustes)
%     Ps  : m×c,    Ps^T Ps = I_c             (OP via M = Σ C_v Z_v^T)
%     Q{v}: m×c,    Q ≥ 0                     (PGD + ReLU)
%     Z{v}: c×n,    col-simplex               (closed-form + simplex)
%     S   : m×n,    col-simplex               (mean + simplex)
%
%   Update order: R → Z → Q → Ps → S
%   Data: X{v} is d_v × n

%% Options
if nargin < 5, opts = struct(); end
max_iter  = get_opt(opts, 'max_iter',  50);
pgd_steps = get_opt(opts, 'pgd_steps', 5);
tol       = get_opt(opts, 'tol',       1e-4);
verbose   = get_opt(opts, 'verbose',   true);

V = length(X); [~, n] = size(X{1}); eps_reg = 1e-6;

if verbose
    fprintf('\n========== BCD-MVRL12 ==========\n');
    fprintf('n=%d, m=%d, c=%d, V=%d, λ=%.4f\n', n, m, c, V, lambda);
    fprintf('R: Procrustes, Ps: OP, Q: PGD+ReLU, Z: closed-form+simplex\n');
    fprintf('Update: R → Z → Q → Ps → S\n');
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
Z=cell(1,V);for v=1:V,Z{v}=proj_simplex(rand(c,n));end
S=proj_simplex(rand(m,n));

obj_old=Inf;obj_history=zeros(max_iter,1);
if verbose,fprintf('\n  Iter   Obj            RelChg\n  ----  --------------  ------\n');end

%% BCD
for iter=1:max_iter
    %% Step 1: R (Orthogonal Procrustes)
    %   B = Ps+Q, M = B Z, X M^T → SVD → R = U V^T
    for v=1:V
        Bv=Ps+Q{v};Mv=Bv*Z{v};
        [U_R,~,V_R]=svd(X{v}*Mv','econ');
        R{v}=U_R*V_R';
    end

    %% Step 2: Z (closed-form + simplex)
    %   (B^T B + λI) Z = B^T R^T X + λ Ps^T S
    for v=1:V
        Bv=Ps+Q{v};
        Z{v}=((Bv'*Bv+lambda*eye(c)+eps_reg*eye(c))\(Bv'*R{v}'*X{v}+lambda*Ps'*S));
        Z{v}=proj_simplex(Z{v});
    end

    %% Step 3: Q (PGD + ReLU, Q ≥ 0)
    %   ∇Q = 2 R^T(R(Ps+Q)Z - X)Z^T
    for v=1:V
        L_Q=2*norm(R{v}'*R{v},2)*norm(Z{v}*Z{v}',2);eta_Q=1/max(L_Q,eps_reg);
        for pgd=1:pgd_steps
            Mv=Ps+Q{v};
            grad_Q=2*R{v}'*(R{v}*Mv*Z{v}-X{v})*Z{v}';
            Q{v}=max(Q{v}-eta_Q*grad_Q,0);
        end
    end

    %% Step 4: Ps (Orthogonal Procrustes via Z Z^T = I_c assumption)
    %   Y_v = R_v^T X_v - Q_v Z_v   [m×n]
    %   C_v = (Y_v + λ S) / (1+λ)   [m×n]
    %   M_Ps = Σ_v C_v Z_v^T        [m×c]
    %   SVD(M_Ps) → Ps = U V^T
    M_Ps=zeros(m,c);
    for v=1:V
        Yv=R{v}'*X{v}-Q{v}*Z{v};
        Cv=(Yv+lambda*S)/(1+lambda);
        M_Ps=M_Ps+Cv*Z{v}';
    end
    [U_P,~,V_P]=svd(M_Ps,'econ');Ps=U_P*V_P';

    %% Step 5: S (mean + simplex)
    S_mean=zeros(m,n);
    for v=1:V,S_mean=S_mean+Ps*Z{v};end
    S=proj_simplex(S_mean/V);

    %% Objective
    obj=0;
    for v=1:V
        Mv=Ps+Q{v};
        obj=obj+norm(X{v}-R{v}*Mv*Z{v},'fro')^2;
        obj=obj+lambda*norm(Ps*Z{v}-S,'fro')^2;
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
