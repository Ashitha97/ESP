#ID Equation

#ASSIGN INPUT

#import sympy as sym

u = float(input("u value: "))

t = float(input("t value: "))
g = float(input("g value: "))
e = float(input("e value: "))
p = float(input("p value: ")) #(row taken as p)
a = float(input("a value: "))
c = float(input("c value: "))
x = float(input("x value: "))



v = (144*e*g/p)**0.5  # Value of v

print("value of v",v)

#v D2/Dx2(u) = D2/Dt2(u) + c D/Dt(u)


#x,t = symbols('x t')

#declaring the expression
#expr = t

#WRT x
#derv_x = diff(expr, x)

# WRT u
#derv_u = diff(expr, t)
              
#print("value of du/dx :")
#display(derv_x)

#print("value of du/dt :")
#display(derv_t)

#x, u = sym.symbols('x u')
#sym.diff(x**2 * u, x)
#p = sym.diff(u, x)
#print("value of p is",p)
#sym.diff

