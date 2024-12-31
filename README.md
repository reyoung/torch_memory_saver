# torch_memory_saver

Allow torch tensor memory to be released and resumed later.

API:

```python
memory_saver = TorchMemorySaver()

# 1. For tensors that wants to be paused, create them within `region`
with memory_saver.region():
    x = torch.full((1_000_000_000,), 100, dtype=torch.uint8, device='cuda')

# 2. After `pause`, CUDA memory is released for those tensors.
# For example, check `nvidia-smi`'s memory usage to verify.
memory_saver.pause()

# 3. After `resume`, CUDA memory is re-occupied for those tensors.
memory_saver.resume()
```

Please refer to https://github.com/sgl-project/sglang/issues/2542#issuecomment-2563641647 for details.

TODO:

- [x] Implementation
- [x] Publish to pypi
- [ ] More tests and infra
- [ ] Documentation
