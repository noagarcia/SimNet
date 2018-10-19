function imdb = load_img_data(delta)
% LOAD_IMG_DATA  Get imdb with image data to train the network

% Load data
imdb = load('data/LandmarksExtra_IMDB_1024.mat');

% Assign new cosine labels
imdb.images.cosines = 2*imdb.images.cosines-1;
cosinesDelta = imdb.images.cosines;
cosinesDelta(imdb.images.label==1) = imdb.images.cosines(imdb.images.label==1) + delta;
cosinesDelta(imdb.images.label==-1) = imdb.images.cosines(imdb.images.label==-1) - delta;
imdb.images.label = cosinesDelta;