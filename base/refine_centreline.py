import numpy as np

__all__ = ['refine']

def refine(a:np.ndarray, n: int = 0) -> np.ndarray:
    """
    Computes iteratively refined curves in metric space using 
    using n steps. Taking in an array of coordinates describing
    a polygonal line, the algorithm identifies the largest segment
    and plugs in the coordinates of its midpoint.
    The algorithm continues until the desired number of refinement 
    steps has been achieved and returns the updated array of 
    coordinates in metric space."""

    # number of refinement steps
    k = 0
    # euclidean distance between successive nodes of the polyline
    spacing = np.linalg.norm(np.diff(a, axis =0), axis =1)

    while k < n:
        
        # array to list.     
        a = [elem for elem in a]

        # identify the segment to be split.
        max_seg_id = np.argmax(spacing)
        new_mid_point = (a[max_seg_id+1] + a[max_seg_id]) / 2
        
        # insert new midpoint coordinates in middle of largest segment
        a.insert(max_seg_id+1, new_mid_point)

        # list to array. 
        a = np.array(a)

        # update the spacing and refinement counter
        spacing = np.linalg.norm(np.diff(a, axis =0), axis =1)
        k+=1
    
    return a