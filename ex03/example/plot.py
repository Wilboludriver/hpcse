from pathlib import Path
import numpy as np
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
import sys
from sklearn.decomposition import PCA


def plot_eigen_values(data_dir, reference_values):

    # Set up plot
    files = list(sorted(data_dir.glob("*_eigen_values.csv")))
    size = 6
    nFiles  = len(files)

    fig, axs = plt.subplots(1, nFiles, figsize=(nFiles * size, size))

    # Set up y scale manually
    ticks = [10 ** i for i in range(3, 10)]
    labels = ["%.1E" % x for x in ticks]

    # Load data from files
    data = []
    for filename in files:
        y = np.loadtxt(filename, delimiter=",")
        y = y.reshape((y.shape[0], -1))
        data.append(y)

    # Set up legend with different colors for each eigen value
    kMax = max([y.shape[1] for y in data])
    colormap = plt.cm.jet([i / kMax for i in range(kMax)])
    
    handles = [plt.plot([],[], color=colormap[i])[0] for i in range(kMax)]
    names = ["$\lambda_{" + str(i + 1) + "}$ = %.1E" % float(reference_values[i]) for i in range(kMax)]
    axs[nFiles // 2].legend(handles, names, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.1), title="Reference Eigenvalues")

    # Plot different methods
    for i, y in enumerate(data):

        # Get shape
        n, k = y.shape

        # Plot evolution of each eigen value
        for j in range(k):
            axs[i].plot(np.arange(n), y[:, j], color=colormap[j])
        
        # Adjust the axis for pretty plotting
        for ax in [axs[i]] if i + 1 < nFiles else [axs[i], axs[i].twinx()]:
            ax.set_yscale("log")
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)
            ax.tick_params(bottom=False, labelbottom=False, left=(i == 0), labelleft=(i == 0))
            ax.set_ylim(ticks[0], ticks[-1])

        # Name the y axis - only for first plot
        if i == 0:
            axs[i].set_ylabel("Eigenvalues of the Covarance Matrix")


        # Set title to name of method
        name = files[i].stem.split("_")[0]
        axs[i].set_title(name[0].upper() + name[1:].lower())

        # Plot grey lines at the individual ticks
        for t in ticks:
            axs[i].axhline(t, 0, n, color="grey", alpha=0.5, zorder=-1)

    # Set figure title
    fig.suptitle("Computing Eigenvalues of the Covarance Matrix\nUsing Different Perceptron Rules", y=1.10, fontsize=15)

    # Save plot
    plt.tight_layout()
    plt.savefig("eigen_values.png", bbox_inches="tight", dpi=300)
    plt.clf()


def plot_eigen_vectors(data_dir, reference_y, shape =(50, 37)):

    # Get relevant files
    files = list(sorted(data_dir.glob("*_eigen_vectors.csv")))
    nFiles  = len(files)

    # Load data from files
    data = []
    for filename in files:
        y = np.loadtxt(filename, delimiter=",").reshape((-1, *shape))
        data.append(y)

    # Find maximal number of eigen vectors
    kMax = max([y.shape[0] for y in data])

    # Set up plot
    size = 6
    fig, axs = plt.subplots(nFiles + 1, kMax, figsize=(nFiles * size, size))

    # Remove axis ticks - we only plot images
    for i in range(nFiles + 1):
        for j in range(kMax):
            axs[i, j].get_xaxis().set_visible(False)
            axs[i, j].spines['top'].set_visible(False)
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].spines['bottom'].set_visible(False)
            axs[i, j].spines['left'].set_visible(False)

            if j != 0:
                axs[i, j].get_yaxis().set_visible(False)
            else:
                axs[i, j].set_yticklabels([])
                axs[i, j].set_xticklabels([])
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])

    # Plot eigen vectors of different methods
    for i, y in enumerate(data):
        
        # Set title to name of method
        name = files[i].stem.split("_")[0]
        axs[i, 0].set_ylabel(name[0].upper() + name[1:].lower())

        # Plot individual eigen vectors
        # Ensuring that the first element is positive
        # allows to fix the eigen values
        for j in range(y.shape[0]):
            axs[i, j].imshow(y[j, 0, 0] * y[j], cmap='gray')

    # Add reference eigen vectors
    axs[nFiles, 0].set_ylabel("Reference Method")
    reference_y = reference_y.reshape((-1, *shape))
    for j in range(kMax):
        axs[nFiles, j].imshow(reference_y[j, 0, 0] * reference_y[j], cmap='gray')

    # Set figure title
    fig.suptitle("Computing Eigenvectors of the Covarance Matrix\nUsing Different Perceptron Rules", fontsize=15)

    # Save plot
    plt.tight_layout()
    plt.savefig("eigen_vectors.png", bbox_inches="tight", dpi=300)
    plt.clf()
        

def get_spectral_decomposition(data, k):

    # Use PCA to compute the eigen vectors and eigen values
    # of the covariance matrix
    pca = PCA(n_components=k, svd_solver='auto').fit(data)

    # The singular values are the square root of the
    # eigen values of X^T * X
    # Hence, we have to scale w.r.t. number of data points
    eigen_values = pca.singular_values_ ** 2 / (data.shape[0] - 1)

    # The eigen vectors equals the prinicpal components
    eigen_vectors = pca.components_

    return eigen_values, eigen_vectors


if __name__ == "__main__":

    # Get parameters
    # data_dir - Directory where data has been stored. Default: output
    # k - Maximal number of eigen values and vectors one whises to compute exactly. Default: 12
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output")
    k =  sys.argv[2] if len(sys.argv) > 2 else 12

    # Compute eigen eigen values and vectors exactly
    data = np.loadtxt("faces.csv", delimiter=",")
    values, vectors = get_spectral_decomposition(data, k)

    # Create plots
    plot_eigen_values(data_dir, values)
