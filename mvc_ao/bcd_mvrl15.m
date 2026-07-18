function [R, Ps, Q, A, w, obj_history] = bcd_mvrl15(X, m, c, lambda, gamma, opts)
% BCD_MVRL15  Multi-View RL (HQ + L1 Hadamard on Ps⊙Q).
%
%   [R, Ps, Q, A, w, obj] = bcd_mvrl15(X, m, c, lambda, gamma, opts)
%
%   Objective:
%     J = Σ_v √(||X^{(v)}-R^{(v)}(Ps+Q^{(v)})A^{(v)}||² + ε)
%       + λ Σ_v ||Ps ⊙ Q^{(v)}||₁                    ← element-wise L1
%       + γ Σ_v ||A^{(v)}||²
%
%   HQ formulation: w_v = 1/(2√(e_v+ε)), weighted least squares
%
%   Constraints:
%     R{v}: d_v×m,  R^T R = I_m              (Procrustes)
%     Ps  : m×c,    Ps^T Ps = I_c             (Procrustes)
%     Q{v}: m×c,    unconstrained             (weighted soft-threshold)
%     A{v}: c×n,    col-simplex               (closed-form + simplex)
%     w_v : scalar, w_v > 0                   (HQ weight)
%
%   NOTE: λ||Ps ⊙ Q||₁ is element-wise L1 on Hadamard product.
%         Q update uses weighted soft-thresholding prox.
%
%   Update order: w → A → R → Ps → Q
%   Data: X{v} is d_v × n

%% Options
if nargin < 6, opts = struct(); end
max_iter = get_opt(opts, 'max_iter', 50);
tol      = get_opt(opts, 'tol',      1e-5);
verbose  = get_opt(opts, 'verbose',  true);

V = length(X); [~, n] = size(X{1}); eps_reg = 1e-8;

if verbose
    fprintf('\n========== BCD-MVRL15 (HQ + L1) ==========\n');
    fprintf('n=%d, m=%d, c=%d, V=%d, λ=%.4f, γ=%.4f\n', n, m, c, V, lambda, gamma);
    fprintf('λ||Ps⊙Q||₁: element-wise L1, Q via weighted soft-threshold\n');
    fprintf('Update: w → A → R → Ps → Q\n');
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
w=ones(V,1);

obj_old=Inf;obj_history=zeros(max_iter,1);
if verbose,fprintf('\n  Iter   Obj            RelChg\n  ----  --------------  ------\n');end

%% BCD
for iter=1:max_iter
    %% Step 1: w (HQ weights)
    for v=1:V
        ev=norm(X{v}-R{v}*(Ps+Q{v})*A{v},'fro')^2;
        w(v)=1/(2*sqrt(max(ev,eps_reg)));
    end

    %% Step 2: A (closed-form + simplex, weighted by w_v)
    %   A = (w_v Z^T Z + γI)^{-1} (w_v Z^T R^T X)
    for v=1:V
        Zv=Ps+Q{v};
        M_A=w(v)*(Zv'*Zv)+gamma*eye(c);
        rhs_A=w(v)*Zv'*R{v}'*X{v};
        A{v}=M_A\rhs_A;
        A{v}=proj_simplex(A{v});
    end

    %% Step 3: R (Orthogonal Procrustes)
    for v=1:V
        Yv=(Ps+Q{v})*A{v};
        [U_R,~,V_R]=svd(X{v}*Yv','econ');
        R{v}=U_R*V_R';
    end

    %% Step 4: Ps (Orthogonal Procrustes, weighted by w_v)
    %   H = Σ_v w_v (R^T X - Q A) A^T
    H_Ps=zeros(m,c);
    for v=1:V
        H_Ps=H_Ps+w(v)*(R{v}'*X{v}-Q{v}*A{v})*A{v}';
    end
    [U_P,~,V_P]=svd(H_Ps,'econ');Ps=U_P*V_P';

    %% Step 5: Q (weighted soft-thresholding)
    %   H_Q = R^T X A^T - Ps A A^T
    %   W = |Ps| (element-wise weight)
    %   Q = sign(H_Q) ⊙ max(|H_Q| - λ·W, 0)
    for v=1:V
        H_Q=R{v}'*X{v}*A{v}'-Ps*A{v}*A{v}';
        W_abs=abs(Ps);
        Q{v}=sign(H_Q).*max(abs(H_Q)-lambda*W_abs,0);
    end

    %% Objective
    obj=0;obj_l1=0;
    for v=1:V
        ev=norm(X{v}-R{v}*(Ps+Q{v})*A{v},'fro')^2;
        obj=obj+sqrt(ev+eps_reg);
        obj=obj+gamma*norm(A{v},'fro')^2;
        obj_l1=obj_l1+sum(sum(abs(Ps.*Q{v})));
    end
    obj=obj+lambda*obj_l1;
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
