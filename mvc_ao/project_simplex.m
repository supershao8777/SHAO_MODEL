function X = project_simplex(Y)
% PROJECT_SIMPLEX  Project each row of Y onto the probability simplex.
%   X = project_simplex(Y) returns X where each row x_i satisfies:
%     x_i >= 0,  sum(x_i) = 1
%
%   Uses the O(n log n) algorithm from:
%   Wang & Carreira-Perpinan (2013), "Projection onto the probability simplex"
%
%   Input:  Y  — n x c matrix
%   Output: X  — n x c matrix, row-stochastic

[n, c] = size(Y);
X = zeros(n, c);

for i = 1:n
    y = Y(i, :)';
    % Sort in descending order
    [ys, idx] = sort(y, 'descend');
    % Find optimal threshold
    cs = cumsum(ys);
    rho = find(ys - (cs - 1) ./ (1:c)', 1, 'last');
    if isempty(rho)
        rho = 1;
    end
    theta = (cs(rho) - 1) / rho;
    % Project
    x = max(y - theta, 0);
    X(i, :) = x';
end
end
