import numpy as np

def calc_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array([point1.x, point1.y]) - np.array([point2.x, point2.y]))
