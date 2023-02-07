# Meta-Reinforcement Learning Algorithms
A PyTorch implementation of meta-reinforcement learning algorithms, RL^2 PPO, SNAIL, and VariBAD. 

TODO

## Setup
Install the following packages.

```bash
# using conda,
conda create --name metarl --file requirements.txt
# Or using pip,
pip install requirements.txt
```

## Usage

TODO

## Algorithms
All base learners use PPO.

- [x] RL^2 Proximal Policy Optimization (PPO)
- [ ] VariBAD
- [ ] SNAIL

Ideas for:
- Proximal Policy Optimization with Episodic Planning Networks (EPNs)


## Results

TODO

## References
- Achiam, J. (2018). Spinning Up in Deep Reinforcement Learning. https://spinningup.openai.com/en/latest/index.html
- Wang, J. X., Kurth-Nelson, Z., Tirumala, D., Soyer, H., Leibo, J. Z., Munos, R., Blundell, C., Kumaran, D., & Botvinick, M. (2017). Learning to reinforcement learn. ArXiv:1611.05763 [Cs, Stat]. http://arxiv.org/abs/1611.05763
- Duan, Y., Schulman, J., Chen, X., Bartlett, P. L., Sutskever, I., & Abbeel, P. (2016). RL$^2$: Fast Reinforcement Learning via Slow Reinforcement Learning (arXiv:1611.02779). arXiv. https://doi.org/10.48550/arXiv.1611.02779
- Zintgraf, L., Shiarlis, K., Igl, M., Schulze, S., Gal, Y., Hofmann, K., & Whiteson, S. (2020). VariBAD: A Very Good Method for Bayes-Adaptive Deep RL via Meta-Learning (arXiv:1910.08348). arXiv. https://doi.org/10.48550/arXiv.1910.08348
- Mishra, N., Rohaninejad, M., Chen, X., & Abbeel, P. (2018). A Simple Neural Attentive Meta-Learner (arXiv:1707.03141). arXiv. http://arxiv.org/abs/1707.03141