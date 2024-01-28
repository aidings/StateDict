# StateDict
> load pytorch's state dict

## install
`pip install git+https://github.com/aidings/StateDict.git`

## examples
```python
from StateDict import StateDict


p = StateDict([torch.nn.Module])

p.load([str], strict=False)


```
