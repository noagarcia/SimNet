# SimNet: similarity network for image retrieval

Matlab code for the paper "Learning Non-Metric Visual Similarity for Image Retrieval", in which a similarity network is proposed to estimate a visual similarity score for ranking images in visual retrieval problems.

![info](https://github.com/noagarcia/SimNet/blob/master/info/SimNet.png?raw=true)


## Prerequisits

- [Matlab](https://www.mathworks.com/products/matlab.html) R2015b on Linux
- [MatConvNet](http://www.vlfeat.org/matconvnet/) 1.0-beta20


## Datasets

In our experiments, we use the following image retrieval datasets:
- [Oxford](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/)
- [Paris](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/)
- [Landmarks](http://sites.skoltech.ru/compvision/projects/neuralcodes/)


## Usage

### Training
1. Download training data from:
- [LandmarksExtra]()
- [LandmarksExtra500]()
- [Landmarks]()

2. In ```load_img_data.m``` file, set path to the training data accordingly.

3. Run ```train.m``` to train the model.

Alternatively, to use your own data, create the ```images``` struct with the fields:
- ```images.data [4-D single]``` (1,D,2,N) array with the pairs of feature vectors, where D is the dimensionality of the feature vectors and N the number of pairs in the dataset. 
- ```images.label [4-D single]``` (1,1,1,N) array with the label of each pair, with 1 being a mathcing pair and -1 being a non-matching pair
- ```images.set [Nx1 double]``` N dimensional vector with the set of each pair, with 1 being training set and 2 being validation set.
- ```images.cosines [4-D single]``` (1,1,1,N) array with the cosine similarity between the pair of feature vectors.

### Test
For Oxford and Paris datasets:
1. Save datasets as:
- ```datasets/{ox,pa}/Images``` with the dataset images (without queries)
- ```datasets/{ox,pa}/Queries``` with the dataset queries (cropped and renamed with their corresponding id).

2. Run ```test.m```

3. Accuracy can be measured with the algorithms provided by the datasets.

## Results

Results obtained with the LandmarksExtra training set:


| Method     | Oxford | Paris |
| ------------ | -------- | ------ |
| Cosine | 0.665 | 0.638 |
| OASIS |  0.619 | 0.853 |
| Linear | 0.602 | 0.581 |
| SimNet | 0.718 | 0.757 |
| SimNet* | 0.808 | 0.891 |


## Citation

```
@article{Garcia2017Learning,
       author    = {Noa Garcia and George Vogiatzis},
       title     = {Learning Non-Metric Visual Similarity for Image Retrieval},
       journal   = {arXiv preprint arXiv:1709.01353},
       year      = {2017},
}
``` 



