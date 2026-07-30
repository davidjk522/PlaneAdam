#utility functions
import torch


def simple_cosine_similarity(moving_im: torch.Tensor, fixed_im: torch.Tensor) -> float:
    """
    Compute the cosine similarity between two flattened feature vectors.
    """
    return torch.nn.functional.cosine_similarity(
        moving_im.flatten(), fixed_im.flatten(), dim=0
    ).item()


def extract_features(model, data_loader, device):
    """
    Extract features from the model for the given data loader.
    Args:
        model: The neural network model.
        data_loader: DataLoader providing input data.
        device: Device to perform computation on (CPU or GPU).
    Returns:
        features: Extracted features from the model. For ViT, shape (B, P, C); for CNN, shape (B, D, C).
        Two tensors Fm(Feature_moving), Ff(Feature_fixed), representing the extracted features for moving and fixed images.
        
    """
    features_moving = []
    features_fixed = []
    for batch in data_loader:
        moving_images = batch['moving'].to(device)
        fixed_images = batch['fixed'].to(device)

        # extract_feature_planecycle already sets eval mode and runs under no_grad.
        moving_features = model.extract_feature_planecycle(moving_images)
        fixed_features = model.extract_feature_planecycle(fixed_images)

        features_moving.append(moving_features)
        features_fixed.append(fixed_features)

    # Concatenate all features along the batch dimension
    features_moving = torch.cat(features_moving, dim=0)
    features_fixed = torch.cat(features_fixed, dim=0)

    return features_moving, features_fixed

def compute_cosine_similarity(moving_features, fixed_features) -> float:
    """
    Compute the cosine similarity between two feature tensors.
    Args:
        moving_features: Tensor of shape (B, D, H, W, C) for moving features.
        fixed_features: Tensor of shape (B, D, H, W, C) for fixed features.
    Returns:
        cosine_similarity: Average cosine similarity across the batch.
    """
    B = moving_features.shape[0]
    cosine_similarities = []
    for b in range(B):
        moving_flat = moving_features[b].flatten()
        fixed_flat = fixed_features[b].flatten()
        cosine_sim = simple_cosine_similarity(moving_flat, fixed_flat)
        cosine_similarities.append(cosine_sim)
    
    return sum(cosine_similarities) / B
