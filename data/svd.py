import numpy as np

# this method computes the singular value decomposition of an input matrix
def compute_svd(img, k=20):
    # convert image array to float
    A = np.squeeze(img.astype(float))
    # get image dimensions
    m, n = A.shape
    # compute right singular vectors and the singular values
    S, V = compute_right_singular_vals(A)
    # compute left singular vectors
    U = compute_left_singular_vals(A, S, V, m, n)
    S = np.nan_to_num(S)
    # reconstruct input using rank k best approximation
    compressed_img = reconstruct_img(U, S, V, k)
    return compressed_img[np.newaxis, :]

# This method computes the right singular vectors (v) and singular values 6t7of the input matrix A
def compute_right_singular_vals(A):
    ATA = A.T @ A
    eigvals, eigvecs = np.linalg.eigh(ATA)
    V = np.flip(eigvecs, 1)
    S = np.sqrt(np.flip(eigvals))
    return S, V

# This method computes the left singular vectors (v) of the input matrix A
def compute_left_singular_vals(A, S, V, m, n):
    U = np.zeros((m,m))
    num_iters = min(m, n)
    for j in range(num_iters):
        U[:, j] = A @ V[:, j] / S[j]
    return U

# this method computes a reconstructe image using the best rank k approximation 
def reconstruct_img(U, S, V, k):
    img = 0
    for j in range(k):
        img += S[j] * np.outer(U[:, j], V[:, j])
    return img