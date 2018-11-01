% Parameters
model_name                      = 'SimNet_delta02_landmarksextra';
delta                           = 0.2;      % delta to learn similarity
do_warmup                       = false;     % true to compute&use warm up weights
use_warmup                      = true;     % true to use warm up weights
dim_feat                        = 512;

trainOpts.gpus                  = [1];      % number of GPUs for training
trainOpts.batchSize             = 100;      % batch size
trainOpts.learningRate          = 0.001;    % learning rate
trainOpts.numEpochs             = 55;      % number of epochs
trainOpts.errorFunction         = 'none' ;

% ***** WARM-UP WEIGHTS *****

if do_warmup 

    % Create some random vectors
    fprintf('Generating random vectos...');
    imdb = imdb_random_vectors(dim_feat, 0.5e6);
    fprintf('...Done\n');
    
    % Initialize the network
    simnet = init_simnet_simplenn(dim_feat);
    simnet = addCustomLossLayer(simnet, @l1LossForward, @l1LossBackward);
    
    % Train
    trainOpts.expDir = 'train/warmup';
    simnet = cnn_train(simnet, imdb, @getBatch, trainOpts);

    % Remove loss layer
    simnet.layers(end) = [];

    % Move the CNN back to the CPU & save for later use
    if trainOpts.gpus > 0
        simnet = vl_simplenn_move(simnet, 'cpu');
    end
    save('models/warmup_weights.mat', '-struct', 'simnet');
    
end

% ***** LEARN SIMILARITY *****

% Clear variables
clear imdb;

% Initialize the network
if do_warmup
    % simnet is alredy computed
    trainOpts.numEpochs = 100;
elseif use_warmup
    % load previous weights
    simnet = load('models/warmup_weights.mat');
    trainOpts.numEpochs = 100;
else
    % ranom weights
    simnet = init_simnet();
    trainOpts.numEpochs = 100;
end
simnet = addCustomLossLayer(simnet, @l1LossForward, @l1LossBackward);

% Image data
imdb = load_img_data(delta);
  
% Train
trainOpts.expDir = 'train/simnet';
simnet = cnn_train(simnet, imdb, @getBatch, trainOpts);
  
% Remove loss layer
simnet.layers(end) = [];
 
% Move the CNN back to the CPU and save for later use
if trainOpts.gpus > 0
    simnet = vl_simplenn_move(simnet, 'cpu');
end
save(sprintf('models/%s.mat', model_name), '-struct', 'simnet');

% Clear variables
clear imdb;


