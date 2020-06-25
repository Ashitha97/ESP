#PRODUCTION RATE
print("PRODUCTION RATE")

spm = float(input("spm value: "))
d = float(input("d value: "))
s = float(input("s value: "))

q = 0.0166*spm*s*d**2
print(" pump production rate is",q)


#HYDRAULIC HP
print("Hydraulic hp eqn")

y = float(input("y value: "))
f = float(input("f value: "))

H = (7.36*10**-6)*q*y*f
print("value of HYDRAULIC HP is",H)


