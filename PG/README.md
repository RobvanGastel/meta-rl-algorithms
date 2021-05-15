### Policy Gradient

The implementation of the vanilla Policy Gradient algorithm (or REINFORCE). Since, the introduction of the REINFORCE algorithms several different baselines have been proposed to reduce the variance and increase the convergence of the algorithm. Several different baselines have been evaluated on the CartPole environment, 

![](../img/result_pg_baselines.png)

Furthermore, the Policy Search problem is described more indepth in the corresponding PDF and the proof of the Policy Gradient Theorem is given.


**References**
- The [deepRL bootcamp lectures](https://www.youtube.com/watch?v=S_gwYj1Q-44) by Peter Abbeel
- The [RLVS](https://rlvs.aniti.fr/) lectures by Olivier Sigaud
- The [blog post](http://karpathy.github.io/2016/05/31/rl/) and [lecture](https://www.youtube.com/watch?v=tqrcjHuNdmQ) by Karpathy
- [REINFORCE](https://link.springer.com/article/10.1007/BF00992696) by Williams
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)