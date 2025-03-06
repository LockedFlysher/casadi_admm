import casadi as ca

A = ca.SX.sym('A',3,3)

B = A[:,ca.Slice(1,2)]

print(A)
print(B)