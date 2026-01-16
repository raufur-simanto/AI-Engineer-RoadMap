# Day 1-2 Quick Start Guide

## 🚀 Getting Started

### 1. Activate Virtual Environment
```bash
cd /home/shimanto/personal_projects/AI-Road-Maps/week1-transformers
source venv/bin/activate
```

### 2. Run the Examples (in order)

#### Day 1 Morning: Attention Mechanism
```bash
python 01_attention_mechanism.py
```
**What you'll learn:**
- How attention works (Q, K, V matrices)
- Scaled dot-product attention
- Self-attention vs cross-attention
- Masked attention for causal models
- Attention visualization

#### Day 1 Afternoon: Multi-Head Attention
```bash
python 02_multi_head_attention.py
```
**What you'll learn:**
- Why multiple attention heads?
- How heads specialize
- Parameter count analysis
- Comparing single vs multi-head

#### Day 2 Morning Part 1: Positional Encoding
```bash
python 03_positional_encoding.py
```
**What you'll learn:**
- Why transformers need positional encoding
- Sinusoidal encoding formula
- Different wavelengths for different dimensions
- Relative position information

#### Day 2 Morning Part 2: Transformer Block
```bash
python 04_transformer_block.py
```
**What you'll learn:**
- Complete encoder block structure
- Residual connections
- Layer normalization
- Feed-forward networks
- Stacking multiple blocks

#### Day 2 Afternoon: Mini Transformer (Training!)
```bash
python 05_mini_transformer.py
```
**What you'll learn:**
- End-to-end transformer
- Training loop implementation
- Sequence reversal task
- Attention pattern visualization
- Model evaluation

---

## 📚 Study Schedule

### Day 1 (4-6 hours)
**Morning (2-3 hours):**
1. Read "The Illustrated Transformer" (30 min)
2. Run `01_attention_mechanism.py` (30 min)
3. Experiment with the code - modify examples (30 min)
4. Run `02_multi_head_attention.py` (30 min)
5. Understand head specialization (30 min)

**Afternoon (2-3 hours):**
1. Watch Andrej Karpathy's "Let's build GPT" (1.5 hours)
2. Take notes on key concepts (30 min)
3. Re-run examples and experiment (1 hour)

### Day 2 (4-6 hours)
**Morning (2-3 hours):**
1. Read "Attention Is All You Need" paper sections 3.1-3.5 (1 hour)
2. Run `03_positional_encoding.py` (30 min)
3. Run `04_transformer_block.py` (30 min)
4. Understand the full architecture (30 min)

**Afternoon (2-3 hours):**
1. Run `05_mini_transformer.py` (30 min)
2. Watch it train - understand the training loop (30 min)
3. Experiment: change hyperparameters and retrain (1 hour)
4. Try to implement a variation (copy task instead of reversal) (1 hour)

---

## 🎯 Learning Checkpoints

After Day 1, you should be able to:
- [ ] Explain attention mechanism to someone
- [ ] Implement scaled dot-product attention from memory
- [ ] Understand Q, K, V matrices
- [ ] Know why we scale by sqrt(d_k)
- [ ] Explain multi-head attention benefits

After Day 2, you should be able to:
- [ ] Explain why positional encoding is needed
- [ ] Understand sinusoidal encoding formula
- [ ] Explain transformer encoder block structure
- [ ] Know what residual connections do
- [ ] Train a simple transformer model

---

## 💡 Tips for Success

1. **Don't just run the code** - Read it line by line
2. **Add print statements** - Understand tensor shapes
3. **Modify and break things** - Best way to learn
4. **Visualize everything** - The code generates plots for a reason
5. **Take notes** - Write down key insights

---

## 🐛 Troubleshooting

### Virtual environment issues:
```bash
# If venv creation fails, install python3-venv first
sudo apt-get install python3-venv python3-full

# Then recreate venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Import errors:
```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

### CUDA/GPU errors:
The code works on CPU by default. If you see CUDA errors, the code will automatically fall back to CPU.

---

## 📊 Expected Outputs

Each script will generate:
- **Console output** with explanations
- **PNG visualizations** saved in the current directory
- **Key insights** printed at the end

### Generated Files:
- `attention_weights.png` - Attention heatmaps
- `multi_head_attention.png` - Multi-head visualizations
- `positional_encoding_heatmap.png` - PE patterns
- `positional_encoding_wavelengths.png` - Different wavelengths
- `training_progress.png` - Training curves
- `transformer_attention_patterns.png` - Learned attention

---

## 🎓 After Completing Day 1-2

You'll be ready for:
- **Day 3-4:** Modern LLM architectures (LLaMA, Mistral)
- **Day 5-7:** Fine-tuning with Hugging Face
- **Week 2:** Advanced fine-tuning (LoRA, QLoRA)

---

## 🔥 Challenge Yourself

After completing all examples, try:

1. **Modify the task:**
   - Change sequence reversal to sequence copying
   - Try sequence sorting
   - Implement simple arithmetic (e.g., [1, 2, 3] → [6])

2. **Experiment with architecture:**
   - Change number of heads (2, 4, 8, 16)
   - Change number of layers (1, 2, 4, 6)
   - Change d_model (64, 128, 256, 512)
   - Observe impact on training

3. **Implement from scratch:**
   - Close all files
   - Try to implement attention mechanism from memory
   - Compare with the provided code

4. **Add features:**
   - Add learning rate scheduling
   - Add early stopping
   - Add validation set
   - Implement beam search for inference

---

**Good luck! 🚀 You're building the foundation for understanding modern LLMs!**
