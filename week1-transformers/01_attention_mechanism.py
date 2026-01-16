"""
Day 1 Morning: Scaled Dot-Product Attention Implementation

This module implements the core attention mechanism from scratch.
Attention allows the model to focus on different parts of the input sequence.

Key Concepts:
- Query (Q): What we're looking for
- Key (K): What we're looking at
- Value (V): What we actually get
- Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    
    Formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    Args:
        d_k: Dimension of key (used for scaling)
    """
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
        
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: Query matrix (batch_size, seq_len, d_k)
            K: Key matrix (batch_size, seq_len, d_k)
            V: Value matrix (batch_size, seq_len, d_v)
            mask: Optional mask (batch_size, seq_len, seq_len)
            
        Returns:
            output: Attention output (batch_size, seq_len, d_v)
            attention_weights: Attention weights (batch_size, seq_len, seq_len)
        """
        # Step 1: Compute attention scores (QK^T)
        # Shape: (batch_size, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Step 2: Scale by sqrt(d_k) to prevent softmax saturation
        scores = scores / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        # Step 3: Apply mask (optional) - used for padding or causal attention
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Step 4: Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Step 5: Apply attention weights to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


def visualize_attention(attention_weights, tokens=None):
    """
    Visualize attention weights as a heatmap
    
    Args:
        attention_weights: Tensor of shape (seq_len, seq_len)
        tokens: Optional list of token strings
    """
    plt.figure(figsize=(10, 8))
    
    # Convert to numpy for plotting
    weights = attention_weights.detach().cpu().numpy()
    
    # Create heatmap
    sns.heatmap(weights, annot=True, fmt='.2f', cmap='viridis',
                xticklabels=tokens if tokens else range(weights.shape[1]),
                yticklabels=tokens if tokens else range(weights.shape[0]))
    
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title('Attention Weights Heatmap')
    plt.tight_layout()
    plt.savefig('attention_weights.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Attention heatmap saved as 'attention_weights.png'")


def example_1_simple_attention():
    """
    Example 1: Simple attention on a small sequence
    """
    print("=" * 60)
    print("Example 1: Simple Attention Mechanism")
    print("=" * 60)
    
    # Hyperparameters
    batch_size = 1
    seq_len = 4
    d_k = 8  # Dimension of queries and keys
    d_v = 8  # Dimension of values
    
    # Create random Q, K, V matrices
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_v)
    
    print(f"\nInput shapes:")
    print(f"Q (Query): {Q.shape}")
    print(f"K (Key): {K.shape}")
    print(f"V (Value): {V.shape}")
    
    # Apply attention
    attention = ScaledDotProductAttention(d_k)
    output, attention_weights = attention(Q, K, V)
    
    print(f"\nOutput shapes:")
    print(f"Output: {output.shape}")
    print(f"Attention weights: {attention_weights.shape}")
    
    print(f"\nAttention weights (should sum to 1 for each query):")
    print(attention_weights[0])
    print(f"\nSum of attention weights per row: {attention_weights[0].sum(dim=-1)}")
    
    # Visualize
    tokens = ['The', 'cat', 'sat', 'down']
    visualize_attention(attention_weights[0], tokens)
    
    return output, attention_weights


def example_2_self_attention():
    """
    Example 2: Self-attention (Q, K, V come from same source)
    """
    print("\n" + "=" * 60)
    print("Example 2: Self-Attention")
    print("=" * 60)
    
    # In self-attention, Q, K, V are all derived from the same input
    batch_size = 1
    seq_len = 5
    d_model = 16
    
    # Simulate input embeddings
    X = torch.randn(batch_size, seq_len, d_model)
    
    # Linear projections to get Q, K, V
    W_q = nn.Linear(d_model, d_model, bias=False)
    W_k = nn.Linear(d_model, d_model, bias=False)
    W_v = nn.Linear(d_model, d_model, bias=False)
    
    Q = W_q(X)
    K = W_k(X)
    V = W_v(X)
    
    print(f"\nInput X shape: {X.shape}")
    print(f"Q, K, V shapes: {Q.shape}")
    
    # Apply self-attention
    attention = ScaledDotProductAttention(d_model)
    output, attention_weights = attention(Q, K, V)
    
    print(f"\nSelf-attention output shape: {output.shape}")
    print(f"\nAttention weights:\n{attention_weights[0]}")
    
    # Visualize
    tokens = ['I', 'love', 'machine', 'learning', '!']
    visualize_attention(attention_weights[0], tokens)
    
    return output, attention_weights


def example_3_masked_attention():
    """
    Example 3: Masked attention (for causal/autoregressive models)
    """
    print("\n" + "=" * 60)
    print("Example 3: Masked (Causal) Attention")
    print("=" * 60)
    
    batch_size = 1
    seq_len = 5
    d_k = 8
    
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)
    
    # Create causal mask (lower triangular matrix)
    # This prevents positions from attending to future positions
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
    
    print(f"\nCausal mask (1 = can attend, 0 = cannot attend):")
    print(mask[0])
    
    # Apply masked attention
    attention = ScaledDotProductAttention(d_k)
    output, attention_weights = attention(Q, K, V, mask)
    
    print(f"\nMasked attention weights:")
    print(attention_weights[0])
    print("\nNotice: Each position can only attend to itself and previous positions!")
    
    # Visualize
    tokens = ['The', 'cat', 'is', 'very', 'cute']
    visualize_attention(attention_weights[0], tokens)
    
    return output, attention_weights


def example_4_attention_interpretation():
    """
    Example 4: Interpreting attention - what does the model focus on?
    """
    print("\n" + "=" * 60)
    print("Example 4: Attention Interpretation")
    print("=" * 60)
    
    # Simulate a sentence: "The quick brown fox jumps"
    tokens = ['The', 'quick', 'brown', 'fox', 'jumps']
    seq_len = len(tokens)
    d_model = 16
    
    # Create embeddings (in reality, these would be learned)
    X = torch.randn(1, seq_len, d_model)
    
    # Make "fox" and "jumps" more similar (simulate semantic relationship)
    X[0, 3] = X[0, 4] + torch.randn(d_model) * 0.1
    
    # Self-attention
    W_q = nn.Linear(d_model, d_model, bias=False)
    W_k = nn.Linear(d_model, d_model, bias=False)
    W_v = nn.Linear(d_model, d_model, bias=False)
    
    Q = W_q(X)
    K = W_k(X)
    V = W_v(X)
    
    attention = ScaledDotProductAttention(d_model)
    output, attention_weights = attention(Q, K, V)
    
    print("\nAttention weights for each word:")
    for i, token in enumerate(tokens):
        print(f"\n'{token}' attends to:")
        weights = attention_weights[0, i].detach().numpy()
        for j, (other_token, weight) in enumerate(zip(tokens, weights)):
            print(f"  {other_token}: {weight:.3f}")
    
    # Visualize
    visualize_attention(attention_weights[0], tokens)
    
    return output, attention_weights


def main():
    """
    Run all examples
    """
    print("\n🚀 Scaled Dot-Product Attention Implementation\n")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run examples
    example_1_simple_attention()
    example_2_self_attention()
    example_3_masked_attention()
    example_4_attention_interpretation()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)
    print("\n📝 Key Takeaways:")
    print("1. Attention allows the model to focus on relevant parts of input")
    print("2. Scaling by sqrt(d_k) prevents softmax saturation")
    print("3. Masks enable causal attention (for GPT-like models)")
    print("4. Attention weights are interpretable - we can see what the model focuses on")
    print("\n🎯 Next: Implement Multi-Head Attention (02_multi_head_attention.py)")


if __name__ == "__main__":
    main()
