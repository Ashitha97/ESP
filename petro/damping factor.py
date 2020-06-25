#DAMPING FACTOR
print("damping factor")

hp = float(input("hp value: "))
t = float(input("t value: "))
pi = float(input("pi value: "))
ai = float(input("ai value: "))
li = float(input("li value: "))
s = float(input("s value: "))
g = float(input("g value: "))
h =  float(input("h value"))


                    
d =(550*144*g*hp-h*t**2)/(2**0.5*3.14**pi*ai*li*s**2)

print(" Damping factor is",d)

