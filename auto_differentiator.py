# 导入 CasADi 库
import casadi as ca

# 创建符号变量
x = ca.SX.sym('x')
y = ca.SX.sym('y')
la=ca.SX.sym('lambda')

# 定义函数 f(x,y) = x² + y²
function = x*x + la*y*y

# 创建变量向量
variables = ca.vertcat(x, y)

# 计算梯度
gradient = ca.gradient(function, variables)

# 打印梯度
print("梯度向量 ∇f = ", gradient)
# symvar可以提取出变量的名称
print(ca.symvar(function))
