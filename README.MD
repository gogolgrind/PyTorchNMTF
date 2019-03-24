### TorchNMTF

A collection of Nonnegative Matrix and Tensor Factorizations models with *further* applications to Multi-modal Data Analysis and Blind Source Separation and Sensor Fusion Problems, implemented with Pytorch.

### Requirements

- PyTorch == 1.0.0 and it's dependences
- skimage
- Pandas
- [ipypb](https://github.com/Evfro/ipypb)

### Usage

```
    see examples.ipynb
```

Here is a result of the inverse transform of a factorized image
![Chelsea the cat.](https://pp.userapi.com/c856132/v856132226/5a0f/fo90ZyRSXeY.jpg)


### Refereces

1. Andrzej Cichocki Rafal Zdunek Anh Huy Phan Shun‐Ichi Amari, Nonnegative Matrix and Tensor Factorizations

2. https://perso.telecom-paristech.fr/essid/teach/NMF_tutorial_ICME-2014.pdf

3. http://pmelchior.net/blog/proximal-matrix-factorization-in-pytorch.html

## TODO

- [x] Basic NMF model with 2 solvers
- [ ] ALS, HALS solvers, and advanced objectives
- [ ] More Examples
- [ ] Nonegative Tensor Factorisation
- [ ] Tests
- [ ] Setup.py
- [ ] Nonrandom initialization.