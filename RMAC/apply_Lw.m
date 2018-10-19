function X = apply_Lw(X, m, P, dimensions)

if nargin < 4
  dimensions = size(P,2);
end

X = P(1:dimensions,:) * bsxfun(@minus,X,m);
l = sqrt(sum(X.^2));
X = bsxfun(@rdivide,X,l);
