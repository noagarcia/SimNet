function dx = l1LossBackward(x,r,p)
% TODO: Replace the following line with your implementation
dx = p * sign(x - r) ;

dx = dx / (size(x,1) * size(x,2)) ;  % normalize by image size
