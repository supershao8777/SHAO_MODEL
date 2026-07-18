function [R, Ps, Q, A, S, w, obj_history] = bcd_mvrl14(X, m, c, lambda, gamma, opts)
% BCD_MVRL14  Multi-View RL with Half-Quadratic (HQ) robust reconstruction.
%
%   [R, Ps, Q, A, S, w, obj] = bcd_mvrl14(X, m, c, lambda, gamma, opts)
%
%   Objective (original):
%     J = Σ_v √(||X^{(v)}-R^{(v)}(Ps+Q^{(v)})A^{(v)}||² + ε)
%       + λ Σ_v ||(Ps+Q^{(v)})A^{(v)} - S||²
%       + γ Σ_v ||A^{(v)}||²
%
%   HQ formulation: w_v = 1/(2√(e_v+ε)), weighted least squares
%
%   Constraints:
%     R{v}: d_v×m,  R^T R = I_m              (Procrustes)
%     Ps  : m×c,    Ps^T Ps = I_c             (Procrustes)
%     Q{v}: m×c,    unconstrained             (right division)
%     A{v}: c×n,    col-simplex               (closed-form + simplex)
%     S   : m×n,    S S^T = I_m               (row-orthogonal, Procrustes)
%     w_v : scalar, w_v > 0                   (HQ weight)
%
%   Update order: w → S → A → R → Ps → Q
%   Data: X{v} is d_v × n

%% Options
if nargin < 6, opts = struct(); end
max_iter = get_opt(opts, 'max_iter', 50);
tol      = get_opt(opts, 'tol',      1e-5);
verbose  = get_opt(opts, 'verbose',  true);

V = length(X); [~, n] = size(X{1}); eps_reg = 1e-8;

if verbose
    fprintf('\n========== BCD-MVRL14 (HQ) ==========\n');
    fprintf('n=%d, m=%d, c=%d, V=%d, λ=%.4f, γ=%.4f\n', n, m, c, V, lambda, gamma);
    fprintf('HQ: w_v = 1/(2√(e_v+ε)), robust reconstruction\n');
    fprintf('Update: w → S → A → R → Ps → Q\n');
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
[S_tmp,~]=qr(randn(n,m),0);S=S_tmp';  % S S^T = I_m
w=ones(V,1);  % HQ weights

obj_old=Inf;obj_history=zeros(max_iter,1);
if verbose,fprintf('\n  Iter   Obj            RelChg\n  ----  --------------  ------\n');end

%% BCD
for iter=1:max_iter
    %% Step 1: w (HQ weights)
    %   e_v = ||X - R(Ps+Q)A||²,  w_v = 1 / (2√(e_v+ε))
    for v=1:V
        ev=norm(X{v}-R{v}*(Ps+Q{v})*A{v},'fro')^2;
        w(v)=1/(2*sqrt(max(ev,eps_reg)));
    end

    %% Step 2: S (Orthogonal Procrustes: S S^T = I_m)
    F_S=zeros(m,n);
    for v=1:V,F_S=F_S+(Ps+Q{v})*A{v};end
    F_S=F_S/V;
    [U_S,~,V_S]=svd(F_S,'econ');S=U_S*V_S';           % S S^T = I_m

    %% Step 3: A (closed-form + simplex, weighted by w_v)
    %   A = ((w_v+λ) Z^T Z + γI)^{-1} (w_v Z^T R^T X + λ Z^T S)
    for v=1:V
        Zv=Ps+Q{v};
        M_A=(w(v)+lambda)*(Zv'*Zv)+gamma*eye(c);
        rhs_A=w(v)*Zv'*R{v}'*X{v}+lambda*Zv'*S;
        A{v}=M_A\(rhs_A+eps_reg*eye(c,n));
        A{v}=proj_simplex(A{v});
    end

    %% Step 4: R (Orthogonal Procrustes)
    for v=1:V
        M_R=X{v}*A{v}'*(Ps+Q{v})';                  % d_v × m
        [U_R,~,V_R]=svd(M_R,'econ');
        R{v}=U_R*V_R';
    end

    %% Step 5: Ps (Orthogonal Procrustes, weighted by w_v)
    %   H = Σ [w_v R^T X A^T + λ S A^T - (w_v+λ) Q A A^T]
    H_Ps=zeros(m,c);
    for v=1:V
        H_Ps=H_Ps+w(v)*R{v}'*X{v}*A{v}'+lambda*S*A{v}'...
             -(w(v)+lambda)*Q{v}*A{v}*A{v}';
    end
    [U_P,~,V_P]=svd(H_Ps,'econ');Ps=U_P*V_P';

    %% Step 6: Q (right division, weighted by w_v)
    %   G = w_v R^T X A^T + λ S A^T - (w_v+λ) Ps A A^T
    %   D = (w_v+λ) A A^T,  Q = G / D
    for v=1:V
        G_Q=w(v)*R{v}'*X{v}*A{v}'+lambda*S*A{v}'...
            -(w(v)+lambda)*Ps*A{v}*A{v}';
        D_Q=(w(v)+lambda)*A{v}*A{v}';
        Q{v}=G_Q/(D_Q+eps_reg*eye(c));
    end

    %% Objective
    obj=0;
    for v=1:V
        ev=norm(X{v}-R{v}*(Ps+Q{v})*A{v},'fro')^2;
        obj=obj+sqrt(ev+eps_reg);
        obj=obj+lambda*norm((Ps+Q{v})*A{v}-S,'fro')^2;
        obj=obj+gamma*norm(A{v},'fro')^2;
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
