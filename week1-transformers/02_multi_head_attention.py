"""
Day 1 Afternoon: Multi-Head Attention Implementation

Multi-head attention allows the model to attend to information from different
representation subspaces at different positions.

Key Idea: Instead of one attention function, we use multiple "heads" in parallel,
each learning different aspects of the relationships between tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Layer
    
    Instead of performing a single attention function with d_model-dimensional keys,
    values and queries, we project them h times with different learned linear projections.
    
    Args:
        d_model: Dimension of model (e.g., 512)
        num_heads: Number of attention heads (e.g., 8)
        dropout: Dropout probability
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k)
        
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()
        # Reshape to (batch_size, seq_len, num_heads, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        # Transpose to (batch_size, num_heads, seq_len, d_k)
        return x.transpose(1, 2)
    
    def combine_heads(self, x):
        """
        Combine heads back to original shape
        
        Args:
            x: (batch_size, num_heads, seq_len, d_k)
        Returns:
            (batch_size, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.size()
        # Transpose to (batch_size, seq_len, num_heads, d_k)
        x = x.transpose(1, 2)
        # Reshape to (batch_size, seq_len, d_model)
        return x.contiguous().view(batch_size, seq_len, self.d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Compute scaled dot-product attention
        
        Args:
            Q, K, V: (batch_size, num_heads, seq_len, d_k)
            mask: Optional mask
        Returns:
            output: (batch_size, num_heads, seq_len, d_k)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, Q, K, V, mask=None):
        """
        Forward pass
        
        Args:
            Q, K, V: (batch_size, seq_len, d_model)
            mask: Optional mask
        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size = Q.size(0)
        
        # 1. Linear projections
        Q = self.W_q(Q)  # (batch_size, seq_len, d_model)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # 2. Split into multiple heads
        Q = self.split_heads(Q)  # (batch_size, num_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # 3. Apply attention
        attn_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 4. Combine heads
        attn_output = self.combine_heads(attn_output)  # (batch_size, seq_len, d_model)
        
        # 5. Final linear projection
        output = self.W_o(attn_output)
        
        return output, attention_weights


def visualize_multi_head_attention(attention_weights, tokens=None, num_heads_to_show=4):
    """
    Visualize attention weights from multiple heads
    
    Args:
        attention_weights: (num_heads, seq_len, seq_len)
        tokens: Optional list of token strings
        num_heads_to_show: Number of heads to visualize
    """
    num_heads = min(attention_weights.size(0), num_heads_to_show)
    
    fig, axes = plt.subplots(1, num_heads, figsize=(5 * num_heads, 4))
    if num_heads == 1:
        axes = [axes]
    
    for head_idx in range(num_heads):
        weights = attention_weights[head_idx].detach().cpu().numpy()
        
        sns.heatmap(weights, annot=True, fmt='.2f', cmap='viridis',
                   xticklabels=tokens if tokens else range(weights.shape[1]),
                   yticklabels=tokens if tokens else range(weights.shape[0]),
                   ax=axes[head_idx], cbar=True)
        
        axes[head_idx].set_title(f'Head {head_idx + 1}')
        axes[head_idx].set_xlabel('Key Position')
        axes[head_idx].set_ylabel('Query Position')
    
    plt.tight_layout()
    plt.savefig('multi_head_attention.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Multi-head attention visualization saved as 'multi_head_attention.png'")


def example_1_basic_multi_head():
    """
    Example 1: Basic multi-head attention
    """
    print("=" * 60)
    print("Example 1: Basic Multi-Head Attention")
    print("=" * 60)
    
    # Hyperparameters
    batch_size = 2
    seq_len = 6
    d_model = 512
    num_heads = 8
    
    # Create random input
    X = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\nInput shape: {X.shape}")
    print(f"d_model: {d_model}")
    print(f"num_heads: {num_heads}")
    print(f"d_k (dimension per head): {d_model // num_heads}")
    
    # Create multi-head attention layer
    mha = MultiHeadAttention(d_model, num_heads)
    
    # Apply multi-head attention (self-attention)
    output, attention_weights = mha(X, X, X)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"  - batch_size: {attention_weights.size(0)}")
    print(f"  - num_heads: {attention_weights.size(1)}")
    print(f"  - seq_len x seq_len: {attention_weights.size(2)} x {attention_weights.size(3)}")
    
    # Visualize first sample, multiple heads
    tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    visualize_multi_head_attention(attention_weights[0], tokens, num_heads_to_show=4)
    
    return output, attention_weights


def example_2_compare_heads():
    """
    Example 2: Compare single-head vs multi-head attention
    """
    print("\n" + "=" * 60)
    print("Example 2: Single-Head vs Multi-Head Comparison")
    print("=" * 60)
    
    batch_size = 1
    seq_len = 5
    d_model = 64
    
    X = torch.randn(batch_size, seq_len, d_model)
    
    # Single-head attention (num_heads=1)
    mha_single = MultiHeadAttention(d_model, num_heads=1)
    output_single, attn_single = mha_single(X, X, X)
    
    # Multi-head attention (num_heads=8)
    mha_multi = MultiHeadAttention(d_model, num_heads=8)
    output_multi, attn_multi = mha_multi(X, X, X)
    
    print(f"\nSingle-head attention:")
    print(f"  Output shape: {output_single.shape}")
    print(f"  Attention weights shape: {attn_single.shape}")
    
    print(f"\nMulti-head attention (8 heads):")
    print(f"  Output shape: {output_multi.shape}")
    print(f"  Attention weights shape: {attn_multi.shape}")
    
    print("\n💡 Key Insight:")
    print("Multi-head attention allows the model to attend to different aspects")
    print("of the input simultaneously. Each head can learn different patterns:")
    print("  - Head 1 might focus on syntax")
    print("  - Head 2 might focus on semantics")
    print("  - Head 3 might focus on positional relationships")
    print("  - etc.")
    
    return output_single, output_multi


def example_3_head_specialization():
    """
    Example 3: Demonstrate how different heads can specialize
    """
    print("\n" + "=" * 60)
    print("Example 3: Head Specialization")
    print("=" * 60)
    
    # Simulate a sentence with clear structure
    tokens = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
    seq_len = len(tokens)
    d_model = 128
    num_heads = 4
    
    # Create input embeddings
    X = torch.randn(1, seq_len, d_model)
    
    # Apply multi-head attention
    mha = MultiHeadAttention(d_model, num_heads)
    output, attention_weights = mha(X, X, X)
    
    print(f"\nSentence: {' '.join(tokens)}")
    print(f"\nAnalyzing {num_heads} attention heads...")
    
    # Analyze each head
    for head_idx in range(num_heads):
        head_weights = attention_weights[0, head_idx]
        
        # Find which positions each token attends to most
        max_attention = head_weights.max(dim=-1)
        max_indices = head_weights.argmax(dim=-1)
        
        print(f"\nHead {head_idx + 1} - Top attention patterns:")
        for i, token in enumerate(tokens[:3]):  # Show first 3 tokens
            attended_token = tokens[max_indices[i]]
            attention_score = max_attention.values[i].item()
            print(f"  '{token}' → '{attended_token}' (score: {attention_score:.3f})")
    
    # Visualize all heads
    visualize_multi_head_attention(attention_weights[0], tokens, num_heads_to_show=num_heads)
    
    return output, attention_weights


def example_4_parameter_count():
    """
    Example 4: Understand parameter count in multi-head attention
    """
    print("\n" + "=" * 60)
    print("Example 4: Parameter Count Analysis")
    print("=" * 60)
    
    d_model = 512
    num_heads = 8
    
    mha = MultiHeadAttention(d_model, num_heads)
    
    # Count parameters
    total_params = sum(p.numel() for p in mha.parameters())
    
    print(f"\nMulti-Head Attention Configuration:")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  d_k (per head): {d_model // num_heads}")
    
    print(f"\nParameter breakdown:")
    print(f"  W_q: {d_model} x {d_model} = {d_model * d_model:,} parameters")
    print(f"  W_k: {d_model} x {d_model} = {d_model * d_model:,} parameters")
    print(f"  W_v: {d_model} x {d_model} = {d_model * d_model:,} parameters")
    print(f"  W_o: {d_model} x {d_model} = {d_model * d_model:,} parameters")
    print(f"  Total: {total_params:,} parameters")
    
    # Compare with different head counts
    print(f"\nParameter count is INDEPENDENT of num_heads!")
    for heads in [1, 4, 8, 16]:
        mha_temp = MultiHeadAttention(d_model, heads)
        params = sum(p.numel() for p in mha_temp.parameters())
        print(f"  {heads} heads: {params:,} parameters")
    
    return mha


def main():
    """
    Run all examples
    """
    print("\n🚀 Multi-Head Attention Implementation\n")
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run examples
    example_1_basic_multi_head()
    example_2_compare_heads()
    example_3_head_specialization()
    example_4_parameter_count()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)
    print("\n📝 Key Takeaways:")
    print("1. Multi-head attention uses multiple attention 'heads' in parallel")
    print("2. Each head can learn different relationships/patterns")
    print("3. Heads operate on lower-dimensional subspaces (d_k = d_model / num_heads)")
    print("4. Parameter count is independent of number of heads")
    print("5. More heads = more diverse attention patterns")
    print("\n🎯 Next: Implement Positional Encoding (03_positional_encoding.py)")


if __name__ == "__main__":
    main()
