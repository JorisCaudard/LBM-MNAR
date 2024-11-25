import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
import os


def plot_data_LBM(data, row_clusters, col_clusters, ax, title):

    # Sort rows and columns by their cluster labels
    sorted_row_indices = np.argsort(row_clusters)  # Sort row indices by cluster labels
    sorted_col_indices = np.argsort(col_clusters)  # Sort column indices by cluster labels

    # Create a sorted version of the matrix
    sorted_data = data[sorted_row_indices][:, sorted_col_indices]

    # Plot data
    im = ax.imshow(sorted_data, cmap='binary')
    ax.set_title(title)
    ax.axis('off')

    # Get the boundaries between clusters
    row_boundaries = np.where(np.diff(row_clusters[sorted_row_indices]))[0]
    col_boundaries = np.where(np.diff(col_clusters[sorted_col_indices]))[0]

    # Add red lines for row boundaries
    for boundary in row_boundaries:
        ax.axhline(boundary + 0.5, color='red', linewidth=2)

    # Add red lines for column boundaries
    for boundary in col_boundaries:
        ax.axvline(boundary + 0.5, color='red', linewidth=2)

    # Add the colorbar on the axis
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_ticks([-1, 0, 1])  # Set tick positions
    cbar.set_ticklabels(['0', 'NA', '1'])  # Set tick labels

    return im

def generate_mcar_data(data, missing_proba=0.3):
    M = np.random.rand(*data.shape) < missing_proba
    mcar_data = data.copy()
    mcar_data[M] = 0

    return mcar_data

def generate_mar_data(data):
    M = np.zeros_like(data, dtype=bool)
    for i in range(n):
        for j in range(m):
            M[i,j] = np.random.rand() > expit(mu + A[i] + C[j])

    mar_data = data.copy()
    mar_data[M] = 0

    return mar_data

def generate_mnar_data(data):
    M = np.zeros_like(data, dtype=bool)
    for i in range(n):
        for j in range(m):
            P_ij = mu + A[i] + B[i] + C[j] + D[j] if data[i,j] == 1 else mu + A[i] - B[i] + C[j] - D[j]
            M[i,j] = np.random.rand() > expit(P_ij)

    mar_data = data.copy()
    mar_data[M] = 0

    return mar_data

def calculate_missing_proportions(data):
    
    return np.sum(data == 0) / data.size



if __name__ == '__main__':

    np.random.seed(42)

    # Cluster parameters
    n = m = 100
    K = 3
    L = 3

    # CLuster parameters
    alpha = beta = [1/3, 1/3, 1/3]
    epsilon = 0.05
    pi = np.array([[epsilon, epsilon, 1 - epsilon],
                   [epsilon, 1-epsilon, 1-epsilon],
                   [1-epsilon, 1-epsilon, epsilon]])


    # MNAR parameters
    mu = 1
    sigma_A = sigma_B = sigma_C = sigma_D =1

    # Generate row/col clusters
    row_clusters = np.random.choice(K, size=n, p=alpha)  # Row clusters
    col_clusters = np.random.choice(L, size=m, p=beta)  # Column clusters

    # Generate latent variable
    A = np.random.normal(0, sigma_A, size=n)
    B = np.random.normal(0, sigma_B, size=n)
    C = np.random.normal(0, sigma_C, size=m)
    D = np.random.normal(0, sigma_D, size=m)

    # Fill X matrix
    X = np.zeros((n,m))

    for i in range(n):
        for j in range(m):
            k = row_clusters[i]
            l = col_clusters[j]
            X[i,j] = 1 if np.random.rand() < pi[k,l] else -1

    X_mcar = generate_mcar_data(X)
    X_mar = generate_mar_data(X)
    X_mnar = generate_mnar_data(X)

    fig, axs = plt.subplots(2, 2, figsize=(20,20))
    plot_data_LBM(X, row_clusters, col_clusters, axs[0,0], f"Original Data, Missing proportion = {calculate_missing_proportions(X):.2%}")
    plot_data_LBM(X_mcar, row_clusters, col_clusters, axs[0,1], f"MCAR Data, Missing proportion = {calculate_missing_proportions(X_mcar):.2%}")
    plot_data_LBM(X_mar, row_clusters, col_clusters, axs[1,0], f"MAR Data, Missing proportion = {calculate_missing_proportions(X_mar):.2%}")
    plot_data_LBM(X_mnar, row_clusters, col_clusters, axs[1,1], f"MNAR Data, Missing proportion = {calculate_missing_proportions(X_mnar):.2%}")

    plt.show()

    DATA_FOLDER = "synth_dataset"
    np.savetxt(os.path.join(DATA_FOLDER, "X.csv"), X, delimiter=";", fmt ="%d")
    np.savetxt(os.path.join(DATA_FOLDER, "X_mcar.csv"), X_mcar, delimiter=";", fmt ="%d")
    np.savetxt(os.path.join(DATA_FOLDER, "X_mar.csv"), X_mar, delimiter=";", fmt ="%d")
    np.savetxt(os.path.join(DATA_FOLDER, "X_mnar.csv"), X_mnar, delimiter=";", fmt ="%d")