"""
Day 2 Afternoon: Complete Mini Transformer

Build an end-to-end transformer for a simple task: sequence reversal.
This combines everything we've learned:
- Embeddings
- Positional encoding
- Multi-head attention
- Feed-forward networks
- Transformer blocks

Task: Given sequence [1, 2, 3, 4, 5], predict [5, 4, 3, 2, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm


class PositionalEncoding(nn.Module):
    """Positional encoding (from previous file)"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention (from previous file)"""
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
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attn_output = torch.matmul(attention_weights, V)
        attn_output = self.combine_heads(attn_output)
        output = self.W_o(attn_output)
        
        return output, attention_weights


class FeedForward(nn.Module):
    """Feed-forward network"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.dropout(self.linear2(self.dropout(F.relu(self.linear1(x)))))


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x, attention_weights


class MiniTransformer(nn.Module):
    """
    Complete Mini Transformer for Sequence-to-Sequence tasks
    
    Architecture:
        Input → Embedding → Positional Encoding → Transformer Blocks → Output Layer
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        d_ff: Feed-forward hidden dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
    """
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=2, 
                 d_ff=512, max_len=100, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input token indices (batch_size, seq_len)
            mask: Optional attention mask
        Returns:
            logits: (batch_size, seq_len, vocab_size)
            attention_weights: List of attention weights from each layer
        """
        # Embedding + positional encoding
        x = self.embedding(x) * np.sqrt(self.d_model)  # Scale embeddings
        x = self.pos_encoding(x)
        
        # Pass through transformer blocks
        attention_weights_list = []
        for encoder_block in self.encoder_blocks:
            x, attention_weights = encoder_block(x, mask)
            attention_weights_list.append(attention_weights)
        
        # Output layer
        logits = self.output_layer(x)
        
        return logits, attention_weights_list


class SequenceReversalDataset(Dataset):
    """
    Dataset for sequence reversal task
    
    Example: [1, 2, 3, 4, 5] → [5, 4, 3, 2, 1]
    """
    def __init__(self, num_samples=1000, seq_len=10, vocab_size=20):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Generate random sequences
        self.data = []
        for _ in range(num_samples):
            # Random sequence (excluding 0, which we'll use for padding)
            seq = torch.randint(1, vocab_size, (seq_len,))
            # Reversed sequence
            reversed_seq = torch.flip(seq, [0])
            self.data.append((seq, reversed_seq))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


def train_model(model, train_loader, num_epochs=20, lr=0.001):
    """
    Train the transformer model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    accuracies = []
    
    print(f"\nTraining on: {device}")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            
            # Forward pass
            logits, _ = model(src)
            
            # Compute loss
            loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            correct += (predictions == tgt).sum().item()
            total += tgt.numel()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        losses.append(avg_loss)
        accuracies.append(accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")
    
    return losses, accuracies


def visualize_training(losses, accuracies):
    """Visualize training progress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(losses, marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(accuracies, marker='o', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Accuracy')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nTraining progress saved as 'training_progress.png'")


def test_model(model, vocab_size=20, seq_len=10, num_examples=5):
    """
    Test the trained model on new sequences
    """
    device = next(model.parameters()).device
    model.eval()
    
    print("\n" + "=" * 60)
    print("Testing Model on New Sequences")
    print("=" * 60)
    
    with torch.no_grad():
        for i in range(num_examples):
            # Generate random sequence
            seq = torch.randint(1, vocab_size, (1, seq_len)).to(device)
            
            # Get model prediction
            logits, attention_weights = model(seq)
            predictions = logits.argmax(dim=-1)
            
            # Ground truth (reversed)
            target = torch.flip(seq, [1])
            
            # Check if correct
            is_correct = (predictions == target).all().item()
            
            print(f"\nExample {i+1}:")
            print(f"  Input:      {seq[0].cpu().tolist()}")
            print(f"  Predicted:  {predictions[0].cpu().tolist()}")
            print(f"  Target:     {target[0].cpu().tolist()}")
            print(f"  Correct:    {'✓' if is_correct else '✗'}")
    
    return attention_weights


def visualize_attention_patterns(model, vocab_size=20, seq_len=10):
    """
    Visualize attention patterns for a sample sequence
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Generate a sample sequence
    seq = torch.randint(1, vocab_size, (1, seq_len)).to(device)
    
    with torch.no_grad():
        logits, attention_weights_list = model(seq)
    
    # Visualize attention from each layer
    num_layers = len(attention_weights_list)
    
    fig, axes = plt.subplots(1, num_layers, figsize=(6 * num_layers, 5))
    if num_layers == 1:
        axes = [axes]
    
    for layer_idx, attention_weights in enumerate(attention_weights_list):
        # Average across heads and batch
        avg_attention = attention_weights[0].mean(dim=0).cpu().numpy()
        
        sns.heatmap(avg_attention, annot=True, fmt='.2f', cmap='viridis',
                   ax=axes[layer_idx], cbar=True)
        axes[layer_idx].set_title(f'Layer {layer_idx + 1} - Avg Attention')
        axes[layer_idx].set_xlabel('Key Position')
        axes[layer_idx].set_ylabel('Query Position')
    
    plt.tight_layout()
    plt.savefig('transformer_attention_patterns.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nAttention patterns saved as 'transformer_attention_patterns.png'")


def main():
    """
    Main training and evaluation pipeline
    """
    print("\n🚀 Mini Transformer - Sequence Reversal Task\n")
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Hyperparameters
    vocab_size = 20
    seq_len = 10
    d_model = 128
    num_heads = 4
    num_layers = 2
    d_ff = 512
    batch_size = 32
    num_epochs = 20
    lr = 0.001
    
    print("Configuration:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  num_layers: {num_layers}")
    print(f"  d_ff: {d_ff}")
    
    # Create dataset and dataloader
    print("\nCreating dataset...")
    dataset = SequenceReversalDataset(num_samples=1000, seq_len=seq_len, vocab_size=vocab_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    print("Creating model...")
    model = MiniTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_len=100
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train model
    losses, accuracies = train_model(model, train_loader, num_epochs=num_epochs, lr=lr)
    
    # Visualize training
    visualize_training(losses, accuracies)
    
    # Test model
    test_model(model, vocab_size=vocab_size, seq_len=seq_len, num_examples=5)
    
    # Visualize attention patterns
    visualize_attention_patterns(model, vocab_size=vocab_size, seq_len=seq_len)
    
    print("\n" + "=" * 60)
    print("✅ Training and evaluation completed!")
    print("=" * 60)
    print("\n📝 What You've Built:")
    print("  ✓ Complete transformer from scratch")
    print("  ✓ Trained on sequence reversal task")
    print("  ✓ Visualized attention patterns")
    print("  ✓ Achieved high accuracy on simple task")
    
    print("\n🎯 Next Steps:")
    print("  1. Try different tasks (copy, sorting, etc.)")
    print("  2. Experiment with hyperparameters")
    print("  3. Add decoder for full seq2seq")
    print("  4. Move to Day 3-4: Modern LLM Architectures")
    
    print("\n🎉 Congratulations! You've implemented a transformer from scratch!")


if __name__ == "__main__":
    main()
