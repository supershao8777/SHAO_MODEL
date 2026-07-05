function X = matrix_cg(L_op, B, X0, max_iter, tol)
% MATRIX_CG  Conjugate Gradient solver for matrix linear systems.
%   X = matrix_cg(L_op, B, X0, max_iter, tol)
%
%   Solves L(X) = B where L_op is a function handle L_op(X) returning
%   a matrix of the same size as X.
%
%   Inputs:
%     L_op     — function handle, computes L(X) given matrix X
%     B        — c x m right-hand side matrix
%     X0       — c x m initial guess
%     max_iter — max CG iterations (default 50)
%     tol      — relative residual tolerance (default 1e-6)
%
%   Output:
%     X        — c x m solution matrix

if nargin < 5, tol = 1e-6; end
if nargin < 4, max_iter = 50; end

X = X0;
R = B - L_op(X);    % residual matrix
P = R;              % search direction
rr_old = sum(R(:).^2);  % squared Frobenius norm

% Handle zero RHS
if rr_old < eps
    return;
end

for k = 1:max_iter
    LP = L_op(P);
    p_lp = sum(P(:) .* LP(:));
    if abs(p_lp) < eps
        break;
    end
    alpha = rr_old / p_lp;
    X = X + alpha * P;
    R = R - alpha * LP;
    rr_new = sum(R(:).^2);

    % Check convergence
    if sqrt(rr_new) / sqrt(sum(B(:).^2) + eps) < tol
        break;
    end

    beta = rr_new / rr_old;
    P = R + beta * P;
    rr_old = rr_new;
end
end
