# 📓 Google Colab Notebooks - AI Engineer Roadmap

## 🚀 Quick Start

All notebooks are designed to run on **Google Colab** with **free GPU access**!

### How to Use:
1. Click the Colab link for each notebook
2. Click "Copy to Drive" to save your own copy
3. Run cells sequentially (Shift + Enter)
4. Experiment and modify the code!

---

## 📚 Week 1: Transformer Fundamentals

### Day 1-2: Building Transformers from Scratch

#### 🔹 Day 1 Morning: Attention Mechanism
**File:** `week1-transformers/Day1_Attention_Mechanism.ipynb`

**What you'll learn:**
- Scaled dot-product attention implementation
- Q, K, V matrices explained
- Self-attention vs cross-attention
- Masked (causal) attention
- Attention visualization

**Time:** 2-3 hours

---

#### 🔹 Day 1 Afternoon: Multi-Head Attention
**File:** `week1-transformers/Day1_MultiHead_Attention.ipynb`

**What you'll learn:**
- Why multiple attention heads?
- Head specialization
- Implementing multi-head attention
- Parameter count analysis
- Comparing single vs multi-head

**Time:** 2-3 hours

---

#### 🔹 Day 2 Morning: Positional Encoding & Transformer Block
**File:** `week1-transformers/Day2_Positional_Encoding_TransformerBlock.ipynb`

**What you'll learn:**
- Sinusoidal positional encoding
- Why transformers need position information
- Complete transformer encoder block
- Residual connections & layer normalization
- Feed-forward networks

**Time:** 2-3 hours

---

#### 🔹 Day 2 Afternoon: Complete Mini Transformer
**File:** `week1-transformers/Day2_Mini_Transformer_Training.ipynb`

**What you'll learn:**
- End-to-end transformer implementation
- Training loop from scratch
- Sequence reversal task
- Attention pattern visualization
- Model evaluation

**Time:** 2-3 hours

---

## 🎯 Learning Path

### Recommended Order:
1. **Day 1 Morning** → Attention Mechanism (Foundation)
2. **Day 1 Afternoon** → Multi-Head Attention (Extension)
3. **Day 2 Morning** → Positional Encoding + Blocks (Architecture)
4. **Day 2 Afternoon** → Complete Transformer (Integration)

### Study Tips:
- ✅ Run all cells sequentially
- ✅ Read the explanations carefully
- ✅ Experiment with hyperparameters
- ✅ Visualize everything
- ✅ Try the challenge exercises

---

## 📖 Required Reading (Before/During)

### Must Read:
1. [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Jay Alammar
2. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original paper

### Must Watch:
1. [Andrej Karpathy - Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)

---

## 🎓 Success Criteria

After completing Week 1 notebooks, you should be able to:
- [ ] Explain attention mechanism to someone else
- [ ] Implement attention from scratch (without looking)
- [ ] Understand transformer architecture completely
- [ ] Train a simple transformer model
- [ ] Visualize and interpret attention patterns

---

## 🔜 Coming Soon

### Week 2: LLM Fine-tuning
- Day 3-4: Hugging Face Transformers
- Day 5-7: LoRA & QLoRA fine-tuning
- Day 8-10: Advanced techniques (DPO, RLHF)

### Week 3: AI Agents
- LangChain agents
- AutoGen multi-agent systems
- CrewAI frameworks

### Week 4: LLM Evaluation
- Benchmark datasets
- Evaluation metrics
- Custom evaluation pipelines

### Week 5: GPU Optimization & Deployment
- vLLM inference
- Model quantization
- Production deployment

---

## 💡 Tips for Colab

1. **Enable GPU:**
   - Runtime → Change runtime type → GPU (T4)
   - Free tier is sufficient for Week 1

2. **Save Your Work:**
   - File → Save a copy in Drive
   - Download notebooks regularly

3. **Session Limits:**
   - Free Colab sessions timeout after ~12 hours
   - Save checkpoints frequently

4. **Memory Management:**
   - Runtime → Restart runtime (if needed)
   - Clear outputs to save memory

---

## 🐛 Troubleshooting

### Common Issues:

**"Runtime disconnected"**
- Colab free tier has usage limits
- Save your work and reconnect

**"Out of memory"**
- Reduce batch size
- Use smaller models
- Restart runtime

**Import errors**
- Run the installation cell first
- Restart runtime if needed

---

## 📞 Need Help?

- Check the notebook comments
- Read error messages carefully
- Google the error (most are common)
- Experiment and learn from mistakes!

---

**Ready to start? Open the first notebook and let's build! 🚀**
