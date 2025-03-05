```shell
pip3 install casadi
```

## 算法拼图

算法需要依次实现：
- 对偶上升法
- 对偶分解
- ADMM

标准ADMM解决的问题形式是：
$$\min_{x,z} f(x) + g(z) $$ $$\text{subject to} \quad Ax + Bz = c$$
对应的增广拉格朗日函数（缩放形式）是：
$L_\rho(x, z, u) = f(x) + g(z) + \frac{\rho}{2}||Ax + Bz - c + u||_2^2 - \frac{\rho}{2}||u||_2^2$
迭代步骤为：

1. $x_{k+1} := \arg\min_x \left( f(x) + \frac{\rho}{2}||Ax + Bz_k - c + u_k||_2^2 \right)$
2. $z_{k+1} := \arg\min_z \left( g(z) + \frac{\rho}{2}||Ax_{k+1} + Bz - c + u_k||_2^2 \right)$
3. $u_{k+1} := u_k + (Ax_{k+1} + Bz_{k+1} - c)$


1.如何设置子问题？

对于MPC优化问题
- 每一步的状态量和输入量需要不相关
- 

$$x_i$$

2.如何