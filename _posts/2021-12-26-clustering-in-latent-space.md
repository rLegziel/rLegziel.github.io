---
layout: post
title:  "Clustering in Latent Space"
date:   2021-12-26 16:02:19 
categories: AI
usemathjax: true

---
# Clustering in Latent Space


_This post is derived from original research performed for my thesis: 'Leveraging representation learning techniques for user session categorization'._

Suppose we have an unlabeled dataset that doesn't cluster well based on objective measures(we have no ground-truth to gauge accuracy on). What can we do?

Embeddings (or latent space representations) are an alternative representation for input data using linear(PCA) or non-linear transformations (Autoencoders, Variational Autoencoders). The aim for embeddings is to be more compact(in terms of feature space and dimensionality) and discard irrelevent information in order to create a more easily seperable representation of the input data.

All algorithms described above do not require any labeled data and can be used as a stepping stone toward an effective self-supervised structure.

## Algorithms

### PCA
PCA is an effective dimensionality reduction technique that linearly projects the data into latent space. This is done by findign the principal components, which are a set of new, uncorrelated and orthogonal variables that aim to maximize the data variance. Findign these new variables is equivalent to performing eigendecomposition on the covariance (or Gram) matrix and thus can be generalized for non-square matrices by using singular value .
Here we use the sklearn implementation of PCA which indeed uses SVD and also gives us an additional evluation metric in the form of explained variance per principal component.

### Autoencoder(AE)
Multi-layer neural network that is composed by an encoder and a decoder. The encoder takes the original input data and output a latent space representation, where the decoder attempts to reconstruct the latent space representation back to its original form. As you might imagine, this architecture is optimized by minimizing the reconstruction loss (Mean-squared error is used here) of the decoder against the input.
In contrast with PCA, autoencoders allow for non-linear projection based on the choice of activation function in the hidden layer.

![Fig 1:AE Architecture used]({{ "/images/assets/AEArchitecture1.png"  | absolute_url}})

### Variational Autoencoder (VAE)
The VAE uses an autoencoder architecture which constructs the latent space representation using stochastic variational inference. This is done within the encoder, where we assume that even the data isn't gaussian we can still sample using a standard normal distribution. This assumption is done because the neural network can still learn a sufficiently powerful functions to map the input data into the latent space and back(within the decoder).
For this architecture, we use a composite loss function: KL divergence plus the mean of MSE. KL Divergence is the measure of how different two distributions are. MSE is used here isntead of the more common binary cross-entropy due to fact that we formulate our reconstruction loss as regression from the latent space back to the original representation. Moreover, binary cross-entropy loss isn't symmetric; not all output errors are penalized in the same way even when the absolute error is the same. This wasn't suitable for our use-case.

![Fig 2:VAE Architecture used]({{ "/images/assets/VAEArchitecture3.png"  | absolute_url}})

where $$q_{\phi}$$  is the approximation function, $$z$$ is the latent space representation,
$X$ and $X'$ are the input and the output respectively



All algorithms above were used to create a 3-dimensional embedding, so we can visualize it effectively and get some intuition based on visualization


## Data
Without going into too much details about the dataset, I can say that it is small, homogenuous and required domain knowledge to perform feature extraction and filtering. Another constraint was that we wanted to ensure that the resulting embeddings will generalize well towards further unseen.
We had approximately 30 features and 900 instances.


## Clustering Algorithm
In this case, we used Agglomerative Clustering implemented by sklearn and used 5 clusters as baseline


## Metrics

- Calinski-Harabasz Index: The index is the ratio between the sum of inter-cluster variance and intra-cluster variance between all clusters. A higher score is considered better.

- Silhouette Score: Takes into account at the mean distances of each points to all points in its cluster and the nearest cluster. A higher score would indicate that the clusters are more seperated and well-defined.

- Davis-Bouldin Score: Takes into account the average distance between points and their cluster centroid and the distance between the centroids themselves.

Moreover, we cab also use the labels from the clustering as classes to be used by a classification algorithm.
Thus, we can also easily evaluate the classification model recall, precision and accuracy.


## Results

### Classification Scores

| Algorithm   | Accuracy          | Recall             | Precision         |
|-------------|-------------------|--------------------|-------------------|
| Baseline    | 0.302 $\pm$ 0.034 | 0.201 $\pm$ 0.034  | 0.714 $\pm$ 0.205 |   
| PCA         | 0.966 $\pm$ 0.008 | 0.966 $\pm$ 0.008  | 0.974 $\pm$ 0.008 |   
| Autoencoder | 0.966 $\pm$ 0.007 | 0.966 $\pm$ 0.006  | 0.973 $\pm$ 0.007 |   
| VAE         | **0.985** $\pm$ 0.005 | **0.985**  $\pm$ 0.004 | **0.986** $\pm$ 0.005 |


### Clustering Scores

| Algorithm   | Silhouette Score | CH index | DB Score |
|-------------|------------------|----------|----------|
| Baseline    | 0.246            | 134.801  | 1.611    |
| PCA         | 0.505            | 622.649  | 0.917    |
| Autoencoder | **0.601**            | 1348.959 | 0.718    |
| VAE         | 0.521           | **2684.568** | **0.539**    |


## Visualizations

![Fig 3:PCA Visualization]({{ "/images/assets/pcaRotatedPlotCropped.png"  | absolute_url}})

We can see a red-flag in the plot, even though the clustering itself seems reasonable. We can see that the explained variance by each principal component is very small, and even the top 3 components account for only 49% of explained variance.
S
![Fig 4:AE Visualization]({{ "/images/assets/aeRotatedPlotCropped.png"  | absolute_url}})

We can see here that many of the clusters are relatively dispersed and have some points that seem further away from the rest of the cluster of even small clusters within one (The purple and orange clusters are good examples).

![Fig 5:VAE Visualization]({{ "/images/assets/vaeRotatedPlotCropped.png"  | absolute_url}})

This visualization is significantly different. All points seem to be closer together and the whole distribution seem much smoother. Each band on its own seems gaussian and the whole distribution also seems gaussian. Clusters are also closer overall closer together. It seems like the cluster divisions are relatively straight such that every cluster occupies a certain slice of of the embedding space.



