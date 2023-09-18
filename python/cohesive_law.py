import sympy
import math

def traction_separation_envelope(lmbda, epsilon, omega):
  return sympy.Piecewise((lmbda*sympy.exp(1-lmbda), lmbda < 1), ( 1-sympy.Pow(1-sympy.Pow(lmbda*sympy.exp(1-lmbda), epsilon), omega), lmbda >= 1))

def G(D, epsilon, omega):
  return sympy.Piecewise((1, D < 1), (traction_separation_envelope(D, epsilon, omega), D >= 1))

def F(D, epsilon, omega):
  return sympy.Piecewise((1, D < 1), (-1/D*sympy.LambertW(-1/math.e * G(D, epsilon, omega)), D >= 1))

def tau(lmbda, D, epsilon, omega):
  return lmbda * F(D, epsilon, omega) * sympy.exp(1-lmbda*F(D, epsilon, omega))

def bilinear(lmbda, D):
  return lmbda * (6 - D) / ( D * (6 - 1))


x = sympy.symbols('x')
"""
sympy.plot(traction_separation_envelope(x, 1, 1),
           traction_separation_envelope(x, 1.569, 2),
           traction_separation_envelope(x, 2.46, 5),
           traction_separation_envelope(x, 3.185, 10),
           traction_separation_envelope(x, 4.95, 50),
           (x,0,9), adaptive=False)

sympy.plot(F(x, 1, 1),
           F(x, 1.569, 2),
           F(x, 2.46, 5),
           F(x, 3.185, 10),
           F(x, 4.95, 50),
           (x,0,9), adaptive=False)

sympy.plot((traction_separation_envelope(x, 1, 1), (x,0,9)),
           (tau(x, 1.0, 1, 1),(x,0,1.0)),
           (tau(x, 1.5, 1, 1),(x,0,1.5)),
           (tau(x, 2.0, 1, 1),(x,0,2.0)),
           (tau(x, 2.5, 1, 1),(x,0,2.5)),
           (tau(x, 3.0, 1, 1),(x,0,3.0)),
           (tau(x, 4.0, 1, 1),(x,0,4.0)),
           (tau(x, 5.0, 1, 1),(x,0,5.0)),
           (tau(x, 6.0, 1, 1),(x,0,6.0)),
           adaptive=False)

sympy.plot((traction_separation_envelope(x, 1, 1), (x,0,9)),
           (bilinear(x, 1.0),(x,0,1.0)),
           (bilinear(x, 1.5),(x,0,1.5)),
           (bilinear(x, 2.0),(x,0,2.0)),
           (bilinear(x, 2.5),(x,0,2.5)),
           (bilinear(x, 3.0),(x,0,3.0)),
           (bilinear(x, 4.0),(x,0,4.0)),
           (bilinear(x, 5.0),(x,0,5.0)),
           (bilinear(x, 6.0),(x,0,6.0)),
           adaptive=False)
"""
def d(delta, delta_f, delta_0):
  return sympy.Piecewise((0, delta <= delta_0),
                         (delta_f / delta * (delta - delta_0)/(delta_f - delta_0), delta > delta_0))

def t(delta, delta_f, delta_0, k_0):
  return ((1-d(delta, delta_f, delta_0)) * k_0 * delta)

sympy.plot(t(x, 5, 1, 1), (x,0.01,5))

"""
def ortiz(delta, d):
  return (1-d) * x * sympy.exp(1 - x)

def ortiz2(delta, d):
  return (1-d) * ((x-d)-1) * sympy.exp(1 - ((x-d)-1))

sympy.plot(ortiz(x,0), ortiz2(x, .5), (x,0, 8), ylim=[0,1])
"""
#p = sympy.plot(traction_separation_envelope(x), tau(x, 2), (x,0,6), adaptive=False)

#p1 = sympy.plot(G(x), (x, 0, 6), adaptive=False)

#p2 = sympy.plot(F(x), (x, 0, 6), adaptive=False)