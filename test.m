addpath('RMAC/')

% Params
use_gpu                     = true;                     % true/false to GPU
L                           = 3;                        % RMAC num of levels

% Import data
dataset                     = 'ox';                     % choose between
                                                        % 'ox' and 'pa'

pathResult = fullfile('results', dataset);              % path to save the
mkdir(pathResult);                                      % output file

load(sprintf('data/RMAC_%s.mat', dataset), 'x');        % Load pre-computed
                                                        % RMACs for the
                                                        % images in the
                                                        % dataset
                                                        
load('data/PCAmat.mat', 'eigvec', 'eigval', 'Xm');      % Load RMAC matrices

Testdir = sprintf('datasets/%s/Queries/', dataset);     % path to queries dir
Testimages = imageSet(Testdir);                         % queries have been
                                                        % cropped and
                                                        % renamed
                                                        
DBdir = sprintf('datasets/%s/Images/', dataset);        % path to dataset
DBimages = imageSet(DBdir);


% Load networks: featnet and simnet
featnet = load('../nets/imagenet-vgg-verydeep-16.mat'); % featnet is a vgg16
featnet.layers = featnet.layers(1:31);                  % without last fc layers

simnet = load('models/SimNet_delta02_landmarksextra.mat');  % simnet is the
                                                            % similarity network

if use_gpu
    featnet = vl_simplenn_move(featnet, 'gpu');
    simnet = vl_simplenn_move(simnet, 'gpu');
end

% Test: iterate over query images
num_queries = Testimages.Count;
num_dbimages = size(x,1);
scores = zeros(num_dbimages, num_queries);

for iimage = 1:num_queries
    
    % Load query, compute regional vectors and apply PCA-whitening
    fprintf('Processing image %d\n', iimage); 
    im = read(Testimages, iimage);                      
	[rvecs, ~] = rmac_regionvec(im, featnet, L);
	[rvecs] = vecpostproc(rvecs);
	for ivec = 1:size(rvecs,2)                
	    rvecs(:,ivec) = vecpostproc(apply_whiten (rvecs(:,ivec), Xm, eigvec, eigval));
	end
	queryRMACvector = vecpostproc(sum(rvecs, 2));
    
    % Match query RMAC against DB images RMACs
    pair = zeros(1,512,2);
    for idb = 1:num_dbimages
        
        % Prepare input simnet
        pair(:,:,1) = queryRMACvector';
        pair(:,:,2) = x(idb,:);       
        pair = single(pair);
        if use_gpu
            pair_ = gpuArray(pair);
        end    
        
        % Apply simnet
        res = vl_simplenn(simnet, pair_);
        simout = res(end).x(1);
        if use_gpu
            simout = gather(simout);                          
        end
        
        % Save result
        scores(idb,iimage) = simout;
      
    end
    
    % Write sorted results in a file
    [maxscore,indexSort] = sort(scores(:,iimage), 'descend');
    [~,name,~] = fileparts(char(Testimages.ImageLocation(iimage)));
    fileID = fopen(fullfile(pathResult, [name '.txt']),'w');
    for itop = 1:length(indexSort)
        [~,outputName,~] = fileparts(char(DBimages.ImageLocation(indexSort(itop))));
        fprintf(fileID,'%s\n',outputName);
    end
 
end

