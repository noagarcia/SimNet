function net = init_simnet_simplenn(dim)
%INITIALIZECOSINECNN  Initialize a small CNN for cosine similarity
%   NET = INITIALIZECOSINECNN() returns the SimpleNN model NET.

net.meta.inputSize = [1 512 2 1] ;

f=1/100 ;
net.layers = { } ;

net.layers{end+1} = struct(...
  'name', 'dist_fc_1', ...
  'type', 'conv', ...
  'weights', {{f*randn(1,dim,2,4096, 'single'), zeros(1, 4096, 'single')}}, ...
  'pad', 0, ...
  'stride', 1, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;

net.layers{end+1} = struct(...
  'name', 'dist_relu_1', ...
  'type', 'relu') ;


net.layers{end+1} = struct(...
  'name', 'dist_fc_2', ...
  'type', 'conv', ...
  'weights', {{f*randn(1,1,4096,4096, 'single'), zeros(1, 4096, 'single')}}, ...
  'pad', 0, ...
  'stride', 1, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;

net.layers{end+1} = struct(...
  'name', 'dist_relu_2', ...
  'type', 'relu') ;


net.layers{end+1} = struct(...
  'name', 'dist_pred', ...
  'type', 'conv', ...
  'weights', {{f*randn(1,1,4096,1, 'single'), zeros(1, 1, 'single')}}, ...
  'pad', 0, ...
  'stride', 1, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;

% Consolidate the network, fixing any missing option
% in the specification above

net = vl_simplenn_tidy(net) ;
