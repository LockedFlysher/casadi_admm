import casadi as ca

# 创建多个符号变量
x = ca.SX.sym('x',2)
y = ca.SX.sym('y')

print(x,y)

# 创建多个输出表达式
f1 = x**2 + y
f2 = x * y

# 创建函数（多输入多输出）
f = ca.Function('f',
                [x, y],           # 输入列表
                [f1, f2]  )     # 输出列表

# 使用函数
result = f(2.0, 3.0)
print("f1 =", result[0])
print("f2 =", result[1])

# 使用命名参数
result = f([1,1], [3.0])
print("使用命名参数:", result)
