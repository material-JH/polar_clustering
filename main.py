from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import numpy as np

def one_round_clustering(n_clusters, manifold_data):
    if np.shape(manifold_data)[1] > 1000:
        manifold_clustering_result = MiniBatchKMeans(n_clusters=n_clusters).fit(manifold_data)
    else:
        manifold_clustering_result = KMeans(n_clusters=n_clusters).fit(manifold_data)

    labels = manifold_clustering_result.labels_ + 1

    return labels, manifold_clustering_result.cluster_centers_


def get_rotation_matrix(i_v, unit=None):
    # From http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q38
    if unit is None:
        unit = [1.0, 0.0, 0.0]
    # Normalize vector length
    i_v /= np.linalg.norm(i_v)

    # Get axis
    uvw = np.cross(i_v, unit)

    # compute trig values - no need to go through arccos and back
    rcos = np.dot(i_v, unit)
    rsin = np.linalg.norm(uvw)

    # normalize and unpack axis
    if not np.isclose(rsin, 0):
        uvw /= rsin
    u, v, w = uvw

    # Compute rotation matrix - re-expressed to show structure
    return (
            rcos * np.eye(3) +
            rsin * np.array([
        [0, -w, v],
        [w, 0, -u],
        [-v, u, 0]
    ]) +
            (1.0 - rcos) * uvw[:, None] * uvw[None, :]
    )

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
