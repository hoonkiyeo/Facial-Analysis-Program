# Facial Analysis Program

---

## Objective

- The objective of this program is to display a visual representation of the original image and the projected image side-by-side.


## Dataset and Packages

- The program will be using part of the [Yale face dataset](http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html).
- The dataset format is npy format.
- The dataset contains 2414 sample images, each of size 32 × 32.

```python
>>> from scipy.linalg import eigh
>>> import numpy as np
>>> import matplotlib.pyplot as plt
```

## Main Concepts & Steps

### Concepts
- In this facial analysis program, various concepts from linear algebra and pca will be applied.
- Will use n to refer to the number of images (so n = 2414) and d to refer to the number of features for each sample image (so d = 1024 = 32×32).
### Steps
1. Load the dataset
```python
>>> x = np.load(filename)
```
- This should give you a n × d dataset (recall that n = 2414 is the number of images in the dataset and d = 1024 is the number of dimensions of each image). In other words, each row represents an image feature vector.
2. Center the dataset
- <img width="110" alt="Screen Shot 2022-06-21 at 2 39 31 AM" src="https://user-images.githubusercontent.com/69660509/174743729-bb92646b-3b79-445e-9e0a-7e34a39624be.png">
- To center the dataset is simply to subtract the mean μx from each data point xi
3. Find the covariance matrix
- One of the interpretations of PCA is that it is the eigendecomposition of the sample covariance matrix.
- The covariance matrix is defined as below
- <img width="149" alt="Screen Shot 2022-06-21 at 3 29 45 AM" src="https://user-images.githubusercontent.com/69660509/174754200-79e5c381-5eb5-4ea0-9216-e251da0293f2.png">
4. Get the m largest eigenvalues and their eigenvectors
- PCA can be performed by doing an eigendecomposition and taking the eigenvectors corresponding to the largest eigenvalues.
- get_eig function returns the diagonal matrix of eigenvalues first, then the eigenvectors in corresponding columns.
- get_eig_prop function returns the diagonal matrix of eigenvalues that explain more than a certain proportion of the variance and the eigenvectors in corresponding columns.
5. Project the images
- project_image function returns the projection of that image.
```python
def project_image(image, U):
    return (np.transpose(image)@U)@np.transpose(U)
```
7. Visualize
```python
def display_image(orig, proj):
    orig = np.transpose(np.reshape(orig, (32,32)))
    proj = np.transpose(np.reshape(proj, (32,32)))
    
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.set_title("Original")
    ax2.set_title("Projection")
    orig_img = ax1.imshow(orig, aspect = "equal")
    proj_img = ax2.imshow(proj, aspect = "equal")
    
    #set size of the figures
    fig.colorbar(orig_img, ax=ax1)
    fig.colorbar(proj_img, ax=ax2)
    fig.set_size_inches(10, 3.5)

    plt.savefig("output.png")
```
## Run the program
```python
>>> x = load_and_center_dataset('YaleB_32x32.npy') 
>>> S = get_covariance(x)
>>> Lambda, U = get_eig(S, 2)
>>> projection = project_image(x[0], U)
>>> display_image(x[0], projection)
```
- Above is the example of the program implementation of the first image with m=2 (second largest eigenvalues)

![output](https://user-images.githubusercontent.com/69660509/174252869-c16ae3ff-70e6-40cb-b4a5-b76e626ecfa8.png)

- The left subplot is the original image.
- The right subplot is the projected image.