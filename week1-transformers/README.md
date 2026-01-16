# Week 1: Day 1-2 - Transformer Fundamentals

## 🎯 Learning Objectives
By the end of Day 1-2, you will:
- ✅ Understand attention mechanism deeply
- ✅ Implement attention from scratch in PyTorch
- ✅ Build a mini transformer for sequence-to-sequence tasks
- ✅ Understand positional encoding
- ✅ Know the difference between encoder, decoder, and encoder-decoder architectures

---

## 📚 Study Materials (2-3 hours)

### Must Read (in order):
1. **[The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)** by Jay Alammar (30 min)
   - Visual explanation of transformer architecture
   - Focus on: self-attention, multi-head attention, positional encoding

2. **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** - Original Paper (1 hour)
   - Read sections 3.1 (Attention), 3.2 (Multi-Head Attention), 3.5 (Positional Encoding)
   - Don't worry if you don't understand everything - we'll implement it!

3. **[The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)** (1 hour)
   - Line-by-line implementation walkthrough
   - Great reference while coding

### Must Watch:
- **[Andrej Karpathy - Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)** (1.5 hours)
  - Build transformer from scratch
  - Excellent intuition building

---

## 🛠️ Hands-On Projects

### Project Structure:
```
week1-transformers/
├── README.md (this file)
├── 01_attention_mechanism.py
├── 02_multi_head_attention.py
├── 03_positional_encoding.py
├── 04_transformer_block.py
├── 05_mini_transformer.py
└── requirements.txt
```

### Setup:
```bash
cd /home/shimanto/personal_projects/AI-Road-Maps/week1-transformers
pip install -r requirements.txt
```

---

## 📝 Implementation Roadmap

### Day 1 Morning: Attention Mechanism
**File:** `01_attention_mechanism.py`
- Implement scaled dot-product attention
- Visualize attention weights
- Test with simple sequences

### Day 1 Afternoon: Multi-Head Attention
**File:** `02_multi_head_attention.py`
- Implement multi-head attention
- Understand why multiple heads help
- Compare single vs multi-head

### Day 2 Morning: Positional Encoding & Transformer Block
**Files:** `03_positional_encoding.py`, `04_transformer_block.py`
- Implement sinusoidal positional encoding
- Build complete transformer encoder block
- Add layer normalization and feed-forward network

### Day 2 Afternoon: Mini Transformer
**File:** `05_mini_transformer.py`
- Build end-to-end transformer for simple task
- Train on sequence reversal or copy task
- Visualize attention patterns

---

## 🎓 Learning Tips

1. **Type the code yourself** - Don't just copy-paste
2. **Add print statements** - Understand tensor shapes at each step
3. **Experiment** - Change hyperparameters and see what happens
4. **Visualize** - Plot attention weights to build intuition
5. **Debug** - If something breaks, that's where learning happens!

---

## ✅ Success Criteria

By end of Day 2, you should be able to:
- [ ] Explain attention mechanism to someone else
- [ ] Implement attention from scratch without looking at code
- [ ] Understand tensor shapes in transformer (batch, seq_len, d_model)
- [ ] Know what Q, K, V matrices represent
- [ ] Explain why we need positional encoding
- [ ] Run a working mini transformer on a simple task

---

## 🚀 Next Steps

After completing Day 1-2:
- Move to Day 3-4: Modern LLM Architectures
- Start exploring Hugging Face Transformers library
- Load and inspect real models (LLaMA, Mistral)

---

**Let's build! 🔥**
