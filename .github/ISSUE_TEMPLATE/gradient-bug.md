---
name: Incorrect gradient
about: A derivative GradLab computes appears to be wrong
labels: bug, gradients
---

**The expression**

```python
import gradlab as gl
x = gl.Tensor(..., requires_grad=True)
y = ...
y.backward()
```

**Expected gradient, and where it comes from**

(Derivation by hand, a textbook rule, or another framework's output.)

**What GradLab produced**

**gradcheck report**

```python
gl.gradcheck(lambda t: ..., x, raise_exception=False).report()
```

```
paste here
```

**Environment**

- GradLab:
- Python:
- NumPy:
- OS:
