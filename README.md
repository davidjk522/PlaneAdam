Project
- Use PlaneCycle to detect 3D feature map that is a proper lifting of 2D feature maps
- Method 1: we use ConvexAdam to optimize the deformation field to do so.
- Method 2: Utilize VoxelMorph like CNN-based Decoder to create deformation field
    - However, can we concat two large DINO-extracted feature maps naively?
    - To reolve this problem, we might have to take the method applied in FMIR -> Use PCA for reduction and then apply the convolution After on

Development Stage

1. First Extract the two feature maps using DINOV2
2. Process the feature map tensors
3. Use ConvexAdam Method to optimize deformation field


Command Ex

python train.py --model:<name> --dataset:<dataset directory> --epochs:<epoch number> --batch:<batch>  