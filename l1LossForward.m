function y = l1LossForward(x,r)
% TODO: Replace the following line with your implementation
delta = x - r ;
y = sum(abs(delta(:))) ;

y = y / (size(x,1) * size(x,2)) ;  % normalize by image size
