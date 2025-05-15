import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_projection(X, y, title='3D Projection', xlabel='X', ylabel='Y', zlabel='Z', cmap='viridis'):
    """
    Creates a 3D scatter plot of the projected data points, colored by their class.

    Args:
        X (np.ndarray): The projected data points with shape (n_samples, 3)
        y (np.ndarray): The class labels
        title (str): The plot title
        xlabel (str): Label for x-axis
        ylabel (str): Label for y-axis
        zlabel (str): Label for z-axis
        cmap (str): The colormap to use for the scatter plot
    """
    if X.shape[1] != 3:
        raise ValueError("Input data must have exactly 3 dimensions for 3D projection plot")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], 
                        c=y, 
                        cmap=cmap,
                        alpha=0.6)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    
    # Add a color bar
    plt.colorbar(scatter)
    
    # Adjust the viewing angle for better visualization
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.show()