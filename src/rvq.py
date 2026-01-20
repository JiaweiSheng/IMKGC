import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class ResidualVectorQuantizationVanilla(nn.Module):
    """
    Residual Vector Quantization (RVQ) module.
    Quantizes input vectors using multiple codebooks in a cascaded manner,
    where each stage quantizes the residual from the previous stage.

    Args:
        num_codebooks: Number of quantizers/codebooks.
        codebook_size: Number of vectors in each codebook.
        codebook_dim: Dimensionality of each codebook vector.
        commitment_cost: Weight for the commitment loss (used for training stability).
        decay: Exponential moving average decay rate (used for codebook updates).
    """
    def __init__(self, 
                 num_codebooks: int, 
                 codebook_size: int, 
                 codebook_dim: int, 
                 commitment_cost: float = 0.25,
                 decay: float = 0.8):
        super().__init__()
        
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        
        # Shared codebook across all stages
        self.codebook = nn.Embedding(self.codebook_size, codebook_dim)

        # Initialize codebook weights uniformly
        nn.init.uniform_(self.codebook.weight, -1/self.codebook_size, 1/self.codebook_size)
        
    def forward(self, inputs: torch.Tensor):
        """
        Forward pass to compute quantized output.

        Args:
            inputs: Input tensor of shape [B, D] or [B, T, D].
            
        Returns:
            quantized: Quantized output tensor.
            all_indices: List of indices from each codebook stage.
            vq_loss: Total vector quantization loss for training.
        """
        original_shape = inputs.shape
        inputs = inputs.view(-1, self.codebook_dim)  # Flatten to [N, D]
        
        residual = inputs  # Initialize residual
        all_indices = []   # Store indices from each codebook
        quantized_out = 0  # Accumulated quantized output
        vq_loss = 0        # Total VQ loss
        
        for i in range(self.num_codebooks):
            # 1. Compute squared Euclidean distances between residual and codebook vectors
            distances = (
                torch.sum(residual**2, dim=1, keepdim=True)
                - 2 * torch.matmul(residual, self.codebook.weight.t())
                + torch.sum(self.codebook.weight.t()**2, dim=0, keepdim=True)
            )
            
            # 2. Find nearest neighbor indices
            indices = torch.argmin(distances, dim=-1)
            indices_onehot = F.one_hot(indices, self.codebook_size).float()  # Fixed: removed [i]
            
            # 3. Retrieve quantized vectors from codebook
            quantized = torch.matmul(indices_onehot, self.codebook.weight)
            all_indices.append(indices)
            
            # 4. Compute quantization loss during training
            if self.training:
                commitment_loss = F.mse_loss(quantized.detach(), residual)
                codebook_loss = F.mse_loss(quantized, residual.detach())
                vq_loss += codebook_loss + self.commitment_cost * commitment_loss

            # 5. Apply Straight-Through Estimator (STE) to preserve gradients
            quantized = residual + (quantized - residual).detach()
            
            # 6. Update residual for next stage
            residual = residual - quantized
            quantized_out += quantized
        
        # Reshape output to original input shape
        quantized_out = quantized_out.view(*original_shape)
        all_indices = torch.stack(all_indices, dim=-1)
        
        return quantized_out, all_indices, vq_loss

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode input into discrete indices."""
        _, indices, _ = self.forward(inputs)
        return indices
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode discrete indices back into continuous vectors."""
        quantized = torch.zeros(
            indices.shape[0], 
            self.codebook_dim,
            device=indices.device
        )
        
        for i in range(self.num_codebooks):
            idx = indices[:, i]
            quantized += self.codebook(idx)
        return quantized