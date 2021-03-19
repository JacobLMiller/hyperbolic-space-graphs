from numpy import exp, pi, linspace, conj
import matplotlib.pyplot as plt

θ = linspace(0, 2*pi, 200)

def circle(radius, center):
    return center + radius*exp(1j*θ)

def plot_curves(curves):
    for c in curves:
        plt.plot(c.real, c.imag)
    plt.axes().set_aspect(1)
    plt.show()
    plt.close()

def mobius(z, a, b, c, d):
    return (a*z + b)/(c*z + d)

def m(curve):
    new = complex(1,1)
    return mobius(curve, 1, -new, conj(-new), 1)


circles = [circle(1,0),circle(2,0),circle(2,2)]
print(circles[1])
plot_curves(circles)
plot_curves([m(c) for c in circles])
plot_curves([m(c) for c in circles])
plot_curves([m(c) for c in circles])
