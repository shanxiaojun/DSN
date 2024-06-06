from typing import Any, Tuple

import numpy as np
import numpy.typing as npt


def np_softmax(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Calculate softmax.

    Args:
        x (npt.NDArray[Any]): Input vector.

    Returns:
        npt.NDArray[Any]: Vector with softmax applied.
    """
    return np.exp(x) / sum(np.exp(x))


def generate_dirichlet(
    batch_size: int,
    class_id: int,
    scale: Tuple[float, float],
    similarity_matrix: npt.NDArray[Any],
    psi: float,
    max_iter: int,
) -> npt.NDArray[Any]:
    """
    Sample logits from Dirichlet distribution based on vector from similarity matrix of learned classes.

    Args:
        batch_size (int): Number of sampled to generate.
        class_id (int): Class ID.
        scale (Tuple[float, float]): Tuple of scales to be used during computing center of the mass for the Dirichlet
            distribution. First element will be used for the first half of the classes and second for the rest.
        similarity_matrix (npt.NDArray[Any]): Similarity matrix.
        psi (float): Parameter controlling how different sampled logit cant be from original class representation
            vector.
        max_iter (int): Maximal amount of iterations of sampling for one logit.

    Raises:
        Exception: Raised when `max_iter` is exceeded.

    Returns:
        npt.NDArray[Any]: Array of generated logits.
    """
    beta = scale[0] if class_id < similarity_matrix.shape[0] // 2 else scale[1]
    x = []
    alpha = similarity_matrix[class_id, :]
    for j in range(batch_size):
        i = 0
        alpha = (alpha - np.min(alpha)) / (np.max(alpha) - np.min(alpha))
        temp = alpha * beta + 1e-8  # Add small number to prevent from being 0
        while True:
            #print(f'Generate Dirichlet sample {j + 1}/{batch_size} - iter: {i}')
            sample = np.random.dirichlet(temp)
            sample_loss = np.square(np.linalg.norm(sample - np_softmax(alpha)))
            #print(f'Sample loss: {sample_loss}')
            if i > max_iter:
                raise Exception(f'Sample generation did not succeeded! Exceeded {max_iter} iterations!')
            if sample_loss >= psi:
                i += 1
                continue
            x.append(sample)
            break

    return np.array(x)
