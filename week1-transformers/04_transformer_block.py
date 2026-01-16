"""
Day 2 Morning: Transformer Encoder Block

A complete transformer encoder block combines:
1. Multi-head self-attention
2. Add & Norm (residual connection + layer normalization)
3. Feed-forward network
4. Add & Norm again

This is the building block of transformer encoders (like BERT).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    Typically: d_ff = 4 * d_model (expansion factor of 4)
    
    Args:
        d_model: Input/output dimension
        d_ff: Hidden dimension (usually 4 * d_model)
        dropout: Dropout probability
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model)
        """
        # Expand to d_ff
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Project back to d_model
        x = self.linear2(x)
        x = self.dropout(x)
        
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention (copied from previous file for completeness)"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size, num_heads, seq_len, d_k = x.size()
        x = x.transpose(1, 2)
        return x.contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attn_output = torch.matmul(attention_weights, V)
        attn_output = self.combine_heads(attn_output)
        output = self.W_o(attn_output)
        
        return output, attention_weights


class TransformerEncoderBlock(nn.Module):
    """
    Complete Transformer Encoder Block
    
    Architecture:
        Input
          ↓
        Multi-Head Attention
          ↓
        Add & Norm (residual + layer norm)
          ↓
        Feed-Forward Network
          ↓
        Add & Norm
          ↓
        Output
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Multi-head attention with residual connection
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = x + self.dropout(attn_output)  # Residual connection
        x = self.norm1(x)  # Layer normalization
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)  # Residual connection
        x = self.norm2(x)  # Layer normalization
        
        return x, attention_weights


def example_1_single_block():
    """
    Example 1: Single transformer encoder block
    """
    print("=" * 60)
    print("Example 1: Single Transformer Encoder Block")
    print("=" * 60)
    
    # Hyperparameters
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048  # 4 * d_model
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  d_ff: {d_ff}")
    print(f"  Input shape: {x.shape}")
    
    # Create transformer block
    encoder_block = TransformerEncoderBlock(d_model, num_heads, d_ff)
    
    # Forward pass
    output, attention_weights = encoder_block(x)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder_block.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\n💡 Key Components:")
    print("  1. Multi-head attention: Captures relationships between tokens")
    print("  2. Feed-forward network: Processes each position independently")
    print("  3. Residual connections: Helps with gradient flow")
    print("  4. Layer normalization: Stabilizes training")
    
    return output, attention_weights


def example_2_stacked_blocks():
    """
    Example 2: Stack multiple transformer blocks (like BERT/GPT)
    """
    print("\n" + "=" * 60)
    print("Example 2: Stacked Transformer Blocks")
    print("=" * 60)
    
    batch_size = 1
    seq_len = 8
    d_model = 256
    num_heads = 8
    d_ff = 1024
    num_layers = 6  # Like BERT-base
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\nBuilding {num_layers}-layer transformer encoder...")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  d_ff: {d_ff}")
    
    # Create stack of encoder blocks
    encoder_blocks = nn.ModuleList([
        TransformerEncoderBlock(d_model, num_heads, d_ff)
        for _ in range(num_layers)
    ])
    
    # Forward pass through all layers
    current_output = x
    all_attention_weights = []
    
    for i, block in enumerate(encoder_blocks):
        current_output, attn_weights = block(current_output)
        all_attention_weights.append(attn_weights)
        print(f"  Layer {i+1} output shape: {current_output.shape}")
    
    # Count total parameters
    total_params = sum(p.numel() for p in encoder_blocks.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Parameters per layer: {total_params // num_layers:,}")
    
    print("\n💡 This is similar to:")
    print("  - BERT-base: 12 layers, 768 d_model, 12 heads")
    print("  - GPT-2 small: 12 layers, 768 d_model, 12 heads")
    print("  - GPT-3: 96 layers, 12288 d_model, 96 heads")
    
    return current_output, all_attention_weights


def example_3_with_masking():
    """
    Example 3: Using attention masks
    """
    print("\n" + "=" * 60)
    print("Example 3: Transformer Block with Masking")
    print("=" * 60)
    
    batch_size = 1
    seq_len = 6
    d_model = 128
    num_heads = 4
    d_ff = 512
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Create padding mask (simulate that last 2 tokens are padding)
    # Shape: (batch_size, 1, 1, seq_len)
    padding_mask = torch.ones(batch_size, 1, 1, seq_len)
    padding_mask[:, :, :, -2:] = 0  # Mask last 2 positions
    
    print(f"\nInput sequence length: {seq_len}")
    print(f"Padding mask (1=real token, 0=padding):")
    print(f"  {padding_mask[0, 0, 0]}")
    
    # Create transformer block
    encoder_block = TransformerEncoderBlock(d_model, num_heads, d_ff)
    
    # Forward pass with mask
    output, attention_weights = encoder_block(x, mask=padding_mask)
    
    print(f"\nOutput shape: {output.shape}")
    print("\nAttention weights for first head (first token):")
    print(attention_weights[0, 0, 0])
    print("\n💡 Notice: Attention to padded positions should be ~0")
    
    return output, attention_weights


def example_4_residual_connections():
    """
    Example 4: Demonstrate importance of residual connections
    """
    print("\n" + "=" * 60)
    print("Example 4: Residual Connections")
    print("=" * 60)
    
    batch_size = 1
    seq_len = 5
    d_model = 64
    num_heads = 4
    d_ff = 256
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Create block
    encoder_block = TransformerEncoderBlock(d_model, num_heads, d_ff)
    
    # Forward pass
    output, _ = encoder_block(x)
    
    # Compute change from input to output
    change = (output - x).abs().mean().item()
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nAverage absolute change: {change:.4f}")
    
    print("\n💡 Residual Connections:")
    print("  - Allow gradients to flow directly through the network")
    print("  - Enable training of very deep networks (100+ layers)")
    print("  - Output = Input + Transformation(Input)")
    print("  - If transformation is harmful, network can learn to ignore it")
    
    return output


def main():
    """
    Run all examples
    """
    print("\n🚀 Transformer Encoder Block Implementation\n")
    
    # Set random seed
    torch.manual_seed(42)
    
    # Run examples
    example_1_single_block()
    example_2_stacked_blocks()
    example_3_with_masking()
    example_4_residual_connections()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)
    print("\n📝 Key Takeaways:")
    print("1. Transformer block = Attention + FFN + Residuals + LayerNorm")
    print("2. Residual connections enable deep networks")
    print("3. Layer normalization stabilizes training")
    print("4. Feed-forward network processes each position independently")
    print("5. Stack multiple blocks to create powerful models (BERT, GPT)")
    print("\n🎯 Next: Build Complete Mini Transformer (05_mini_transformer.py)")


if __name__ == "__main__":
    main()
