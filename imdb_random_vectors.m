function imdb = imdb_random_vectors(D, N)
% IMDBRANDOMVECTORS  Get imdb of random vectors
%   Inputs: D - vector dimensionality
%           N - number of vectors
%   Ouput:  imdb - structure with random pairs of vectors
%           imdb.images.data - size (1,D,2,N) with N pairs of random vectors
%           imdb.images.label - label for each pair (cosine sim)
%           imdb.images.set - set for each pair (1 val, 2 tain)

data = zeros(1,D,2,N);
C = zeros(1,1,1,N);
for k=1:N;
	x = 2*rand(D,1)-1;
	x=x/norm(x);
	c = 2*rand-1;
	w = randn(D,1);
	u = w-(w'*x)*x;
	u=u/norm(u);
	y=sqrt(1-c^2)*u + c*x;
	C(1,1,1,k)=x'*y;
    img(:,:,1) = x';
    img(:,:,2) = y';
    data(:,:,:,k) = img;
end;

% Assign imdb
imdb.images.data = single(data);
imdb.images.label = single(C);
numpairs = length(imdb.images.label);
imdb.images.set = single(ones(1,numpairs));
imdb.images.set(floor(numpairs*0.75):end) = 2;