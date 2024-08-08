## Llama2.0_inf

this is code from 

https://www.youtube.com/watch?v=oM4VmoabDAI

I follow the video and code this repo



You can download the Llama model by following this link.

https://github.com/meta-llama/llama


when I follow the video I have this issue in these code

```python

        if load_model:
            # The only unmatched key in the checkpoint is rope.freqs. Remove it
            del checkpoint['rope.freqs']
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {time.time() - prev_time:.2f}s")
```

You can try the original code first. If that doesn't work, you can try this one.

```python
checkpoint_filtered = {k: v for k, v in checkpoint.items() if not k.endswith(".rope.freqs")}
model.load_state_dict(checkpoint_filtered, strict=True)

```

Good Luck!
