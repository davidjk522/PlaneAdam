#utility functions



def simple_cosine_similarity(moving_im, fixed_im) -> float:
    """
    Compute the cosine similarity between two tensors
    Moving feature tensor and fixed feature tensor.
    """
    dot_product = sum(x*y for x, y in zip(moving_im, fixed_im))
    norm_a = sum(x**2 for x in moving_im) ** 0.5
    norm_b = sum(y**2 for y in fixed_im) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


