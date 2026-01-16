"""
Day 2 Morning: Positional Encoding Implementation

Transformers have no inherent notion of token order (unlike RNNs).
Positional encoding adds information about the position of tokens in the sequence.

Key Idea: Add position-dependent patterns to embeddings so the model can
distinguish between "cat sat" and "sat cat".
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    where:
        pos = position in sequence
        i = dimension index
        d_model = embedding dimension
    
    Args:
        d_model: Dimension of embeddings
        max_len: Maximum sequence length
        dropout: Dropout probability
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the div_term for the sinusoidal functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        # Register as buffer (not a parameter, but part of the module state)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Add positional encoding to input embeddings
        
        Args:
            x: Input embeddings (batch_size, seq_len, d_model)
        Returns:
            x + positional encoding (batch_size, seq_len, d_model)
        """
        # Add positional encoding (broadcasting handles batch dimension)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def visualize_positional_encoding(pe_matrix, max_len=100, d_model=128):
    """
    Visualize positional encoding as a heatmap
    
    Args:
        pe_matrix: Positional encoding matrix (max_len, d_model)
        max_len: Number of positions to show
        d_model: Number of dimensions to show
    """
    plt.figure(figsize=(15, 8))
    
    # Get the positional encoding values
    pe_values = pe_matrix[:max_len, :d_model].numpy()
    
    # Create heatmap
    sns.heatmap(pe_values, cmap='RdBu', center=0, 
                xticklabels=range(0, d_model, 8),
                yticklabels=range(0, max_len, 10))
    
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Position in Sequence')
    plt.title('Positional Encoding Heatmap')
    plt.tight_layout()
    plt.savefig('positional_encoding_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Positional encoding heatmap saved as 'positional_encoding_heatmap.png'")


def visualize_pe_dimensions(pe_matrix, positions=[0, 1, 10, 50, 100]):
    """
    Visualize how positional encoding varies across dimensions for specific positions
    
    Args:
        pe_matrix: Positional encoding matrix
        positions: List of positions to visualize
    """
    fig, axes = plt.subplots(len(positions), 1, figsize=(12, 3 * len(positions)))
    
    if len(positions) == 1:
        axes = [axes]
    
    for idx, pos in enumerate(positions):
        pe_values = pe_matrix[pos, :].numpy()
        axes[idx].plot(pe_values)
        axes[idx].set_title(f'Positional Encoding at Position {pos}')
        axes[idx].set_xlabel('Dimension')
        axes[idx].set_ylabel('Value')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('positional_encoding_dimensions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Positional encoding dimensions plot saved as 'positional_encoding_dimensions.png'")


def visualize_pe_wavelengths(pe_matrix, dimensions=[0, 1, 10, 50, 100]):
    """
    Visualize how different dimensions have different wavelengths
    
    Args:
        pe_matrix: Positional encoding matrix
        dimensions: List of dimensions to visualize
    """
    plt.figure(figsize=(14, 8))
    
    max_pos = min(200, pe_matrix.size(0))
    
    for dim in dimensions:
        pe_values = pe_matrix[:max_pos, dim].numpy()
        plt.plot(pe_values, label=f'Dimension {dim}')
    
    plt.xlabel('Position')
    plt.ylabel('Encoding Value')
    plt.title('Positional Encoding: Different Dimensions Have Different Wavelengths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('positional_encoding_wavelengths.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Positional encoding wavelengths plot saved as 'positional_encoding_wavelengths.png'")


def example_1_basic_pe():
    """
    Example 1: Basic positional encoding
    """
    print("=" * 60)
    print("Example 1: Basic Positional Encoding")
    print("=" * 60)
    
    d_model = 128
    max_len = 200
    
    # Create positional encoding
    pe_layer = PositionalEncoding(d_model, max_len)
    
    print(f"\nPositional Encoding Configuration:")
    print(f"  d_model: {d_model}")
    print(f"  max_len: {max_len}")
    print(f"  PE matrix shape: {pe_layer.pe.shape}")
    
    # Visualize
    visualize_positional_encoding(pe_layer.pe[0], max_len=100, d_model=d_model)
    
    print("\n💡 Observations:")
    print("1. Each position has a unique encoding pattern")
    print("2. Lower dimensions (left) have longer wavelengths")
    print("3. Higher dimensions (right) have shorter wavelengths")
    print("4. This allows the model to attend to relative positions")
    
    return pe_layer


def example_2_add_to_embeddings():
    """
    Example 2: Adding positional encoding to embeddings
    """
    print("\n" + "=" * 60)
    print("Example 2: Adding PE to Embeddings")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 10
    d_model = 64
    
    # Create random embeddings (simulating word embeddings)
    embeddings = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\nInput embeddings shape: {embeddings.shape}")
    print(f"Sample embedding (first token, first 10 dims):")
    print(embeddings[0, 0, :10])
    
    # Create and apply positional encoding
    pe_layer = PositionalEncoding(d_model, dropout=0.0)  # No dropout for visualization
    embeddings_with_pe = pe_layer(embeddings)
    
    print(f"\nEmbeddings with PE shape: {embeddings_with_pe.shape}")
    print(f"Sample embedding with PE (first token, first 10 dims):")
    print(embeddings_with_pe[0, 0, :10])
    
    # Visualize the difference
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    
    # Original embeddings
    sns.heatmap(embeddings[0].numpy(), cmap='viridis', ax=axes[0], cbar=True)
    axes[0].set_title('Original Embeddings')
    axes[0].set_xlabel('Dimension')
    axes[0].set_ylabel('Position')
    
    # Positional encoding
    sns.heatmap(pe_layer.pe[0, :seq_len].numpy(), cmap='RdBu', center=0, ax=axes[1], cbar=True)
    axes[1].set_title('Positional Encoding')
    axes[1].set_xlabel('Dimension')
    axes[1].set_ylabel('Position')
    
    # Combined
    sns.heatmap(embeddings_with_pe[0].detach().numpy(), cmap='viridis', ax=axes[2], cbar=True)
    axes[2].set_title('Embeddings + PE')
    axes[2].set_xlabel('Dimension')
    axes[2].set_ylabel('Position')
    
    plt.tight_layout()
    plt.savefig('embeddings_with_pe.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nVisualization saved as 'embeddings_with_pe.png'")
    
    return embeddings_with_pe


def example_3_wavelength_analysis():
    """
    Example 3: Analyze wavelengths of different dimensions
    """
    print("\n" + "=" * 60)
    print("Example 3: Wavelength Analysis")
    print("=" * 60)
    
    d_model = 128
    max_len = 500
    
    pe_layer = PositionalEncoding(d_model, max_len)
    
    print("\nAnalyzing wavelengths across dimensions...")
    print("\nKey Insight: Different dimensions encode position at different scales")
    print("  - Low dimensions (0, 1): Long wavelengths → capture distant relationships")
    print("  - High dimensions (126, 127): Short wavelengths → capture local relationships")
    
    # Visualize specific dimensions
    visualize_pe_dimensions(pe_layer.pe[0], positions=[0, 10, 50, 100, 200])
    
    # Visualize wavelengths
    visualize_pe_wavelengths(pe_layer.pe[0], dimensions=[0, 2, 10, 30, 60, 100, 126])
    
    return pe_layer


def example_4_relative_position():
    """
    Example 4: Demonstrate how PE helps with relative positions
    """
    print("\n" + "=" * 60)
    print("Example 4: Relative Position Information")
    print("=" * 60)
    
    d_model = 64
    max_len = 100
    
    pe_layer = PositionalEncoding(d_model, max_len, dropout=0.0)
    
    # Get positional encodings for different positions
    pos_0 = pe_layer.pe[0, 0]
    pos_1 = pe_layer.pe[0, 1]
    pos_10 = pe_layer.pe[0, 10]
    pos_11 = pe_layer.pe[0, 11]
    
    # Compute similarities (dot product)
    sim_0_1 = torch.dot(pos_0, pos_1).item()
    sim_10_11 = torch.dot(pos_10, pos_11).item()
    sim_0_10 = torch.dot(pos_0, pos_10).item()
    
    print("\nSimilarity between positional encodings (dot product):")
    print(f"  Position 0 and 1 (distance=1): {sim_0_1:.4f}")
    print(f"  Position 10 and 11 (distance=1): {sim_10_11:.4f}")
    print(f"  Position 0 and 10 (distance=10): {sim_0_10:.4f}")
    
    print("\n💡 Key Observation:")
    print("Positions with the same relative distance have similar patterns!")
    print("This helps the model learn relative position relationships.")
    
    # Compute similarity matrix
    seq_len = 50
    similarity_matrix = torch.zeros(seq_len, seq_len)
    
    for i in range(seq_len):
        for j in range(seq_len):
            similarity_matrix[i, j] = torch.dot(pe_layer.pe[0, i], pe_layer.pe[0, j])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix.numpy(), cmap='coolwarm', center=0, 
                xticklabels=range(0, seq_len, 5),
                yticklabels=range(0, seq_len, 5))
    plt.xlabel('Position')
    plt.ylabel('Position')
    plt.title('Similarity Between Positional Encodings')
    plt.tight_layout()
    plt.savefig('pe_similarity_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSimilarity matrix saved as 'pe_similarity_matrix.png'")
    
    return similarity_matrix


def main():
    """
    Run all examples
    """
    print("\n🚀 Positional Encoding Implementation\n")
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run examples
    example_1_basic_pe()
    example_2_add_to_embeddings()
    example_3_wavelength_analysis()
    example_4_relative_position()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)
    print("\n📝 Key Takeaways:")
    print("1. Positional encoding adds position information to embeddings")
    print("2. Uses sinusoidal functions with different wavelengths")
    print("3. Different dimensions capture position at different scales")
    print("4. Allows model to learn relative positions")
    print("5. No learnable parameters - fixed mathematical function")
    print("\n🎯 Next: Build Transformer Block (04_transformer_block.py)")


if __name__ == "__main__":
    main()
