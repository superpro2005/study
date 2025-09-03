import numpy as np
import matplotlib.pyplot as plt


F = 100
f0 = 100000
N0 = 1

tau = np.linspace(-0.01,0.01,1000)
R = N0/(np.pi*tau)*np.sin(2*np.pi*F*tau) * np.cos(2*np.pi*f0*tau)


plt.plot (tau,R)
plt.grid(True)
plt.xlabel("t")
plt.ylabel("R(t)")
plt.show()