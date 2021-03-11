import numpy as np

def project_to_rgb(x):
    """
    Args:
        x (nparray(float)): n times g array representing n values with g > 3
            dimensions. We convert this array to an n times 3 array by 
            random projection of the standard basis in g dimensional space
            to three dimensions.
    """
    g = x.shape[1]
    basis = np.eye(g)

    Q, _ = np.linalg.qr(np.random.normal(0, 1, (g, 3)))

    colours = x @ basis @ Q
    colours = colours - np.amin(colours)
    colours = colours / np.amax(colours)
    return colours
