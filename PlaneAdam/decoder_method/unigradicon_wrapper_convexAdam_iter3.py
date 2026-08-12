import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import numpy as np
from scipy.ndimage import median_filter

sys.path.append(os.path.join(os.path.dirname(__file__), 'uniGradICON', 'uniGradICON', 'src'))
from unigradicon import get_unigradicon
import icon_registration as icon
from utils.loss import NccLoss


def MINDSSC(img, radius=1, dilation=2, device=torch.device("cuda")):
    """Extract MIND features"""
    kernel = torch.ones(1, 1, 3, 3, 3).to(device) / 27
    img_pad = F.pad(img, [1, 1, 1, 1, 1, 1], mode='reflect')
    mean_img = F.conv3d(img_pad, kernel, padding=0)
    var_img = F.conv3d(img_pad ** 2, kernel, padding=0) - mean_img ** 2
    var_img = torch.clamp(var_img, min=1e-8)

    mind_features = torch.cat([
        (img - mean_img) / torch.sqrt(var_img + 1e-8),
        img / torch.sqrt(var_img + 1e-8)
    ], dim=1)

    return mind_features


class EnhancedLoss(nn.Module):
    """Improved loss function combining multiple loss terms"""

    def __init__(self, ncc_weight=5.0, mind_weight=1.0, smooth_weight=0.1, jac_weight=0.01):
        super().__init__()
        self.ncc = NccLoss([9, 9, 9])
        self.ncc_weight = ncc_weight
        self.mind_weight = mind_weight
        self.smooth_weight = smooth_weight
        self.jac_weight = jac_weight

    def compute_gradient(self, x):
        """Compute spatial gradient"""
        dx = x[:, :, 1:] - x[:, :, :-1]
        dy = x[:, :, :, 1:] - x[:, :, :, :-1]
        dz = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        return dx, dy, dz

    def compute_jacobian_det(self, flow):
        """Compute Jacobian determinant"""
        # Assume flow shape is [B, 3, H, W, D]
        dx = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
        dy = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
        dz = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]

        # Simplified 3D Jacobian
        jac = 1.0 + dx[:, 0, :, :-1, :-1] * dy[:, 1, :-1, :, :-1] * dz[:, 2, :-1, :-1, :]
        return jac

    def forward(self, fixed, warped, flow, features_fixed=None, features_warped=None):
        losses = {}

        # NCC loss
        ncc_loss = self.ncc(fixed, warped)
        losses['ncc'] = ncc_loss * self.ncc_weight

        # MIND feature loss (if provided)
        if features_fixed is not None and features_warped is not None:
            mind_loss = ((features_warped - features_fixed) ** 2).mean()
            losses['mind'] = mind_loss * self.mind_weight

        # Edge-preserving smoothness regularization
        dx, dy, dz = self.compute_gradient(flow)
        smooth_loss = (dx ** 2).mean() + (dy ** 2).mean() + (dz ** 2).mean()
        losses['smooth'] = smooth_loss * self.smooth_weight

        # Jacobian regularization (prevent folding)
        jac = self.compute_jacobian_det(flow)
        jac_loss = ((jac - 1) ** 2).mean() + (F.relu(-jac + 0.1) ** 2).mean() * 10
        losses['jacobian'] = jac_loss * self.jac_weight

        total_loss = sum(losses.values())

        return total_loss, losses


def create_hybrid_model_with_config(opt):
    """
    Factory function to create UniGradICON_ConvexAdam_Hybrid model with correct configuration
    """
    # Extract configuration from opt dictionary and unknown kwargs
    config = {}

    # Handle known parameters
    config['debug'] = opt.get('debug', False)
    config['lambda_weight'] = float(opt.get('lambda_weight', 0.1))
    config['selected_niter'] = int(opt.get('selected_niter', 10))
    config['use_multiscale'] = opt.get('use_multiscale', True)
    config['enable_post_processing'] = opt.get('enable_post_processing', True)

    # Handle unknown keyword arguments (from command line)
    if 'nkwargs' in opt:
        for key, value in opt['nkwargs'].items():
            try:
                # Try to convert to appropriate type
                if key in ['debug', 'use_multiscale', 'enable_post_processing']:
                    config[key] = value.lower() in ['true', '1', 'yes']
                elif key in ['lambda_weight', 'gaussian_sigma']:
                    config[key] = float(value)
                elif key in ['selected_niter', 'median_filter_size', 'gaussian_kernel_size', 'grid_sp_adam']:
                    config[key] = int(value)
                else:
                    config[key] = value
            except (ValueError, AttributeError):
                print(f"Warning: Could not parse parameter {key}={value}")

    print("Creating UniGradICON_ConvexAdam_Hybrid with configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    return UniGradICON_ConvexAdam_Hybrid(config)


class UniGradICON_ConvexAdam_Hybrid(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {}

        # UniGradICON model
        self.model = get_unigradicon()
        if torch.cuda.is_available():
            self.model = self.model.cuda().float()

        # ConvexAdam parameters
        self.mind_r = config.get('mind_r', 1)
        self.mind_d = config.get('mind_d', 2)
        self.lambda_weight = config.get('lambda_weight', 0.1)
        self.selected_niter = config.get('selected_niter', 10)
        self.selected_smooth = config.get('selected_smooth', 0)
        self.grid_sp_adam = config.get('grid_sp_adam', 1)
        self.debug = config.get('debug', False)
        self.test_time_optimization_time = 0.0
        # Multi-scale optimization parameters
        self.use_multiscale = config.get('use_multiscale', False)
        self.scales = [4, 2, 1]  # Coarse to fine
        self.iterations_per_scale = [10, 10, 5]
        self.learning_rates = [0.01, 0.005, 0.001]

        # Post-processing parameters
        self.enable_post_processing = config.get('enable_post_processing', False)
        self.median_filter_size = config.get('median_filter_size', 3)
        self.gaussian_kernel_size = config.get('gaussian_kernel_size', 3)
        self.gaussian_sigma = config.get('gaussian_sigma', 1.0)

        # Use enhanced loss function
        self.enhanced_loss = EnhancedLoss()

        # Expected input shape for UniGradICON
        self.expected_shape = (175, 175, 175)

        # Print configuration for debugging
        if self.debug:
            print("Debug: UniGradICON_ConvexAdam_Hybrid Configuration:")
            print(f"  - lambda_weight: {self.lambda_weight}")
            print(f"  - selected_niter: {self.selected_niter}")
            print(f"  - use_multiscale: {self.use_multiscale}")
            print(f"  - enable_post_processing: {self.enable_post_processing}")
            print(f"  - debug: {self.debug}")

        # Warmup model
        self._warmup()

    def _warmup(self):
        """Warmup model to initialize cuDNN and dynamic attributes"""
        try:
            with torch.no_grad():
                if torch.cuda.is_available():
                    dummy_input = torch.randn(1, 1, 175, 175, 175, dtype=torch.float32).cuda()
                    self.model.eval()
                    _ = self.model(dummy_input, dummy_input)
                    torch.cuda.empty_cache()
                    print("UniGradICON model warmup successful")
        except Exception as e:
            print(f"Warning: Model warmup failed: {e}")

    def normalize_intensity(self, image):
        """Normalize image intensity to [0,1] range"""
        batch_size = image.shape[0]
        normalized = torch.zeros_like(image)

        for b in range(batch_size):
            img = image[b]
            min_val = img.min()
            max_val = img.max()
            if max_val - min_val > 0:
                normalized[b] = (img - min_val) / (max_val - min_val)
            else:
                normalized[b] = torch.zeros_like(img)

        return normalized

    def adaptive_enhancement_factor(self, flow_magnitude, target_magnitude=0.1):
        """Dynamically compute enhancement factor to avoid over-compensation"""
        if flow_magnitude < 0.01:
            factor = min(2.65, target_magnitude / flow_magnitude)
        elif flow_magnitude < 0.05:
            factor = min(2.0, target_magnitude / flow_magnitude)
        else:
            factor = 1.0

        factor = min(factor, 2.65)  # Upper bound protection

        if self.debug:
            print(f"Debug: Adaptive enhancement factor: {factor:.2f} (flow magnitude: {flow_magnitude:.4f})")

        return factor

    def post_process_flow(self, flow):
        """Post-process to improve flow field quality - COMPLETELY FIXED VERSION"""
        device = flow.device

        if self.debug:
            print(f"Debug: Post-processing flow field with shape: {flow.shape}")

        # Apply median filter more efficiently
        try:
            # Median filter to remove outliers (executed on CPU)
            flow_np = flow.cpu().numpy()
            for i in range(3):
                flow_np[0, i] = median_filter(flow_np[0, i], size=self.median_filter_size)
            flow = torch.from_numpy(flow_np).to(device)
        except Exception as e:
            if self.debug:
                print(f"Warning: Median filter failed: {e}, skipping...")

        # Gaussian smoothing
        kernel = self._gaussian_kernel_3d(self.gaussian_kernel_size, self.gaussian_sigma, device)

        if self.debug:
            print(f"Debug: Gaussian kernel shape: {kernel.shape}")

        flow_smooth = torch.zeros_like(flow)
        for i in range(3):
            # FIXED: Correct tensor dimensions for conv3d
            # flow[:, i:i+1] already has shape [1, 1, 175, 175, 175] (5D)
            # No need to unsqueeze(0) which would make it 6D
            channel_flow = flow[:, i:i + 1]  # Shape: [1, 1, 175, 175, 175]

            if self.debug and i == 0:
                print(f"Debug: Channel flow shape: {channel_flow.shape}")

            # Apply 3D convolution - expects 5D input: [batch, channels, depth, height, width]
            convolved = F.conv3d(
                channel_flow,  # 5D tensor: [1, 1, 175, 175, 175]
                kernel,  # 5D kernel: [1, 1, 3, 3, 3]
                padding=self.gaussian_kernel_size // 2
            )

            flow_smooth[:, i:i + 1] = convolved

        if self.debug:
            print(f"Debug: Post-processed flow field shape: {flow_smooth.shape}")

        return flow_smooth

    def _gaussian_kernel_3d(self, kernel_size, sigma, device=None):
        """Create 3D Gaussian kernel - FIXED VERSION"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create coordinate grids
        center = kernel_size // 2
        coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - center

        # Create 3D meshgrid
        z, y, x = torch.meshgrid(coords, coords, coords, indexing='ij')

        # Calculate distances
        dist_sq = x ** 2 + y ** 2 + z ** 2

        # Calculate Gaussian kernel values
        kernel_3d = torch.exp(-dist_sq / (2 * sigma ** 2))

        # Reshape to 5D for conv3d: [out_channels, in_channels, depth, height, width]
        kernel = kernel_3d.unsqueeze(0).unsqueeze(0)

        # Normalize
        kernel = kernel / kernel.sum()

        return kernel

    def forward(self, moving, fixed, moving_seg=None, fixed_seg=None,
                moving_sam=None, fixed_sam=None, registration=False):

        if not registration:
            return torch.zeros(moving.shape[0], 3, *moving.shape[2:],
                               dtype=torch.float32, device=moving.device)
        self.test_time_optimization_time = 0.0
        # Ensure input data type is float32
        moving = moving.float().contiguous()
        fixed = fixed.float().contiguous()

        # Preprocessing: intensity normalization
        moving = self.normalize_intensity(moving)
        fixed = self.normalize_intensity(fixed)

        # Save original dimensions and device
        original_shape = moving.shape[2:]
        device = moving.device
        batch_size = moving.shape[0]

        # Resize images to match UniGradICON's expected input
        if moving.shape[2:] != self.expected_shape:
            moving_resized = F.interpolate(moving, size=self.expected_shape,
                                           mode='trilinear', align_corners=False)
            fixed_resized = F.interpolate(fixed, size=self.expected_shape,
                                          mode='trilinear', align_corners=False)
            current_shape = self.expected_shape
        else:
            moving_resized = moving
            fixed_resized = fixed
            current_shape = original_shape

        # Ensure resized tensors are contiguous
        moving_resized = moving_resized.contiguous()
        fixed_resized = fixed_resized.contiguous()

        # Stage 1: Get initial deformation field using UniGradICON
        initial_flow = self._get_unigrad_initial_flow(moving_resized, fixed_resized, current_shape)

        # Assess initial quality
        initial_quality = self._assess_quality(moving_resized, fixed_resized, initial_flow)
        if self.debug:
            print(f"Debug: Initial registration quality score: {initial_quality:.3f}")

        # Stage 2: Decide optimization strategy based on quality
        if self.selected_niter > 0:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            optimization_start = time.time()

            if initial_quality > 0.8:
                # Quality is already good, minimal optimization
                if self.debug:
                    print("Debug: Initial quality is good, performing quick optimization")
                optimized_flow = self._convexadam_optimization(
                    moving_resized, fixed_resized, initial_flow, current_shape,
                    iterations=min(10, self.selected_niter)
                )
            elif self.use_multiscale:
                # Use multi-scale optimization
                if self.debug:
                    print("Debug: Using multi-scale optimization strategy")
                optimized_flow = self._multiscale_optimization(
                    moving_resized, fixed_resized, initial_flow, current_shape
                )
            else:
                # Standard optimization
                optimized_flow = self._convexadam_optimization(
                    moving_resized, fixed_resized, initial_flow, current_shape
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            optimization_end = time.time()
            self.test_time_optimization_time = optimization_end - optimization_start
            if self.debug:
                print(f"Debug: Test-time optimization took {self.test_time_optimization_time:.4f} seconds")
        else:
            optimized_flow = initial_flow
            self.test_time_optimization_time = 0.0

        # Post-processing (optional)
        if self.enable_post_processing:
            optimized_flow = self.post_process_flow(optimized_flow)
        elif self.debug:
            print("Debug: Post-processing disabled")

        # Resize flow field back to original dimensions
        if original_shape != current_shape:
            optimized_flow = self._resize_flow(optimized_flow, original_shape, current_shape, device)

        # Ensure output flow field shape is correct
        if optimized_flow.dim() == 4:
            optimized_flow = optimized_flow.unsqueeze(0)

        if optimized_flow.shape[0] != batch_size:
            optimized_flow = optimized_flow.repeat(batch_size, 1, 1, 1, 1)

        if self.debug:
            print(f"Debug: Final output flow field shape: {optimized_flow.shape}")

        return optimized_flow.contiguous()

    def _assess_quality(self, moving, fixed, flow):
        """Assess registration quality"""
        with torch.no_grad():
            # Compute warped image
            grid = F.affine_grid(torch.eye(3, 4).unsqueeze(0).to(flow.device),
                                 flow.shape, align_corners=False)
            warped = F.grid_sample(moving, grid + flow.permute(0, 2, 3, 4, 1),
                                   align_corners=False)

            # Compute NCC similarity
            ncc = -NccLoss([9, 9, 9])(fixed, warped).item()

            # Compute flow field smoothness
            dx = (flow[:, :, 1:] - flow[:, :, :-1]).abs().mean()
            dy = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean()
            dz = (flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]).abs().mean()
            smoothness = 1.0 / (1.0 + dx + dy + dz)

            quality = 0.7 * ncc + 0.3 * smoothness.item()
            return quality

    def _get_unigrad_initial_flow(self, moving, fixed, current_shape):
        """Get initial deformation field using UniGradICON"""
        self.model.eval()
        with torch.no_grad():
            _ = self.model(moving, fixed)

            phi_AB = self.model.phi_AB_vectorfield
            identity = self.model.identity_map

            if identity.shape[0] == 1 and phi_AB.shape[0] > 1:
                identity = identity.repeat(phi_AB.shape[0], 1, 1, 1, 1)

            flow_normalized = phi_AB - identity
            flow = self._normalized_to_voxel_flow(flow_normalized, current_shape)


        return flow


    def _multiscale_optimization(self, moving, fixed, initial_flow, current_shape):
        """Multi-scale optimization strategy - FIXED VERSION"""
        device = moving.device
        current_flow = initial_flow

        for scale, iters, lr in zip(self.scales, self.iterations_per_scale, self.learning_rates):
            if self.debug:
                print(f"Debug: Optimizing at scale 1/{scale}, {iters} iterations, learning rate {lr}")

            # Downsample
            scale_factor = 1.0 / scale
            size_scaled = tuple(int(s * scale_factor) for s in current_shape)

            moving_scaled = F.interpolate(moving, size=size_scaled, mode='trilinear', align_corners=False)
            fixed_scaled = F.interpolate(fixed, size=size_scaled, mode='trilinear', align_corners=False)
            flow_scaled = F.interpolate(current_flow, size=size_scaled, mode='trilinear', align_corners=False)
            flow_scaled = flow_scaled * scale_factor

            # Optimize at current scale
            flow_optimized = self._convexadam_optimization(
                moving_scaled, fixed_scaled, flow_scaled, size_scaled,
                iterations=iters, learning_rate=lr
            )

            # FIXED: Ensure flow_optimized has correct dimensionality
            if flow_optimized.dim() == 4:
                flow_optimized = flow_optimized.unsqueeze(0)

            # Upsample back to original resolution
            current_flow = F.interpolate(flow_optimized, size=current_shape,
                                         mode='trilinear', align_corners=False)
            current_flow = current_flow * scale

        return current_flow

    def _convexadam_optimization(self, moving, fixed, initial_flow, current_shape,
                                 iterations=None, learning_rate=None):
        """Improved ConvexAdam optimization - FIXED VERSION"""
        device = moving.device
        H, W, D = current_shape

        if iterations is None:
            iterations = self.selected_niter
        if learning_rate is None:
            learning_rate = 0.005

        # Force enable gradients
        with torch.enable_grad():
            moving_for_grad = moving.requires_grad_(True)
            fixed_for_grad = fixed.detach()

            # Downsample initial flow field to Adam optimization resolution
            initial_flow_lr = F.interpolate(
                initial_flow.detach(),
                size=(H // self.grid_sp_adam, W // self.grid_sp_adam, D // self.grid_sp_adam),
                mode='trilinear',
                align_corners=False
            )

            # Direct parameter optimization
            displacement_field = nn.Parameter(
                (initial_flow_lr.squeeze(0).permute(1, 2, 3, 0) / self.grid_sp_adam).contiguous()
            )

            # Use Adam optimizer with weight decay
            optimizer = torch.optim.Adam([displacement_field], lr=learning_rate, weight_decay=1e-4)

            # Create base grid
            with torch.no_grad():
                grid0 = F.affine_grid(
                    torch.eye(3, 4).unsqueeze(0).to(device),
                    (1, 1, H // self.grid_sp_adam, W // self.grid_sp_adam, D // self.grid_sp_adam),
                    align_corners=False
                ).squeeze(0)

                scale = torch.tensor([
                    (H // self.grid_sp_adam - 1) / 2,
                    (W // self.grid_sp_adam - 1) / 2,
                    (D // self.grid_sp_adam - 1) / 2
                ]).to(device)

            best_loss = float('inf')
            best_disp = None

            # Pre-compute fixed image features
            with torch.no_grad():
                patch_image_fix = F.avg_pool3d(fixed_for_grad, self.grid_sp_adam, stride=self.grid_sp_adam)
                features_fix = MINDSSC(fixed_for_grad, self.mind_r, self.mind_d, device)
                patch_features_fix = F.avg_pool3d(features_fix, self.grid_sp_adam, stride=self.grid_sp_adam)

            # Optimization loop
            for iter in range(iterations):
                optimizer.zero_grad()

                # Regularization loss
                reg_loss = self.lambda_weight * (
                        ((displacement_field[1:, :, :] - displacement_field[:-1, :, :]) ** 2).mean() +
                        ((displacement_field[:, 1:, :] - displacement_field[:, :-1, :]) ** 2).mean() +
                        ((displacement_field[:, :, 1:] - displacement_field[:, :, :-1]) ** 2).mean()
                )

                # Compute deformation grid
                grid_disp = grid0 + (displacement_field / scale).flip(-1)

                # Downsample moving image
                patch_image_mov = F.avg_pool3d(moving_for_grad, self.grid_sp_adam, stride=self.grid_sp_adam)

                # Warp image
                warped_image = F.grid_sample(
                    patch_image_mov,
                    grid_disp.unsqueeze(0),
                    align_corners=False,
                    mode='bilinear'
                )

                # Compute moving image features
                features_mov = MINDSSC(moving_for_grad, self.mind_r, self.mind_d, device)
                patch_features_mov = F.avg_pool3d(features_mov, self.grid_sp_adam, stride=self.grid_sp_adam)

                # Warp features
                patch_mov_sampled = F.grid_sample(
                    patch_features_mov,
                    grid_disp.unsqueeze(0),
                    align_corners=False,
                    mode='bilinear'
                )

                # Use enhanced loss function
                fitted_grid_temp = displacement_field.permute(3, 0, 1, 2).unsqueeze(0) * self.grid_sp_adam
                total_loss, loss_dict = self.enhanced_loss(
                    patch_image_fix, warped_image, fitted_grid_temp,
                    patch_features_fix, patch_mov_sampled
                )
                total_loss = total_loss + reg_loss

                # Check gradients
                if not total_loss.requires_grad:
                    if self.debug:
                        print(f"Warning: Missing gradients at iteration {iter}")
                    break

                # Backpropagation
                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_([displacement_field], max_norm=1.0)

                optimizer.step()

                # Save best result
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_disp = displacement_field.data.clone()

                if iter % 10 == 0 and self.debug:
                    grad_norm = displacement_field.grad.norm().item() if displacement_field.grad is not None else 0
                    print(
                        f"ConvexAdam iter {iter}/{iterations}, loss: {total_loss.item():.6f}, grad_norm: {grad_norm:.6f}")

            # Use best weights
            if best_disp is not None:
                displacement_field.data = best_disp

            # Generate final high-resolution displacement field
            with torch.no_grad():
                fitted_grid = displacement_field.permute(3, 0, 1, 2).unsqueeze(0) * self.grid_sp_adam

                disp_hr = F.interpolate(
                    fitted_grid,
                    size=(H, W, D),
                    mode='trilinear',
                    align_corners=False
                )

                if self.debug:
                    print(f"Debug: ConvexAdam optimized flow field statistics")
                    print(f"  Flow field shape: {disp_hr.shape}")
                    print(f"  Average magnitude: {disp_hr.abs().mean().item():.2f}")
                    print(f"  Maximum magnitude: {disp_hr.abs().max().item():.2f}")

        # FIXED: Return tensor with correct batch dimension
        return disp_hr  # Keep the batch dimension instead of squeezing

    def _resize_flow(self, flow, target_shape, source_shape, device):
        """Resize flow field to target dimensions"""
        if self.debug:
            print(f"Debug: Resizing flow field from {source_shape} to {target_shape}")
            print(f"Debug: Input flow field shape: {flow.shape}")

        needs_unsqueeze = False
        if flow.dim() == 4:
            flow = flow.unsqueeze(0)
            needs_unsqueeze = True

        flow_resized = F.interpolate(flow, size=target_shape,
                                     mode='trilinear', align_corners=False)

        for i in range(3):
            scale = target_shape[i] / source_shape[i]
            flow_resized[:, i] = flow_resized[:, i] * scale

        if needs_unsqueeze:
            flow_resized = flow_resized.squeeze(0)

        if self.debug:
            print(f"Debug: Output flow field shape: {flow_resized.shape}")

        return flow_resized