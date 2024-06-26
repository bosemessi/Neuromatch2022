import numpy as np
import matplotlib.pyplot as plt

t_max = 150e-3   # second
dt = 1e-3        # second
tau = 20e-3      # second
el = -60e-3      # milivolt
vr = -70e-3      # milivolt
vth = -50e-3     # milivolt
r = 100e6        # ohm
i_mean = 25e-11  # ampere

# print(t_max, dt, tau, el, vr, vth, r, i_mean)

# t = np.arange(0, 0.010, 0.001)
# I_t = i_mean * (1 + np.sin(2*np.pi*t/0.01))

# for i in range(len(t)):
#     print(f"{t[i]:.3f}  {I_t[i]:.4e}")

# plt.plot(t, I_t)
# plt.show()

#################################################
## TODO for students: fill out compute v code ##
# Fill out code and comment or remove the next line
# raise NotImplementedError("Student exercise: You need to fill out code to compute v")
#################################################

# Initialize step_end and v0

step_end = 10
v = el

# Loop for step_end steps
for step in range(step_end):
  # Compute value of t
  t = step * dt

  # Compute value of i at this time step
  i = i_mean * (1 + np.sin((t * 2 * np.pi) / 0.01))

  # Compute v
  v = v + dt * (el - v + r*i)/tau

  # Print value of t and v
  print(f"{t:.3f} {v:.4e}")

#################################################
## TODO for students: fill out the figure initialization and plotting code below ##
# Fill out code and comment or remove the next line
# raise NotImplementedError("Student exercise: You need to fill out current figure code")
#################################################

# Initialize step_end
step_end = 25

# Initialize the figure
fig, ax = plt.subplots(figsize = (5,5))
ax.set_title("I(t)")
ax.set_xlabel("t")
ax.set_ylabel("I(t)")

# Loop for step_end steps
for step in range(step_end):

  # Compute value of t
  t = step * dt

  # Compute value of i at this time step
  i = i_mean * (1 + np.sin((t * 2 * np.pi) / 0.01))

  # Plot i (use 'ko' to get small black dots (short for color='k' and marker = 'o'))
  ax.plot(t, i, "ko")

# Display the plot
plt.show()

#################################################
## TODO for students: fill out the figure initialization and plotting code below ##
# Fill out code and comment or remove the next line
# raise NotImplementedError("Student exercise: You need to fill out membrane potential figure code")
#################################################

# Initialize step_end
step_end = int(t_max / dt)

# Initialize v0
v = el

# Initialize the figure
plt.figure()
plt.title('$V_m$ with sinusoidal I(t)')
plt.xlabel('time (s)')
plt.ylabel('$V_m$ (V)');

# Loop for step_end steps
for step in range(step_end):

  # Compute value of t
  t = step * dt

  # Compute value of i at this time step
  i = i_mean * (1 + np.sin((t * 2 * np.pi) / 0.01))

  # Compute v
  v = v + dt/tau * (el - v + r*i)

  # Plot v (using 'k.' to get even smaller markers)
  plt.plot(t, v, 'k.')

# Display plot
plt.show()


#################################################
## TODO for students: fill out code to get random input ##
# Fill out code and comment or remove the next line
# raise NotImplementedError("Student exercise: You need to fill out random input code")
#################################################

# Set random number generator
np.random.seed(2020)

# Initialize step_end and v
step_end = int(t_max / dt)
v = el

# Initialize the figure
plt.figure()
plt.title('$V_m$ with random I(t)')
plt.xlabel('time (s)')
plt.ylabel('$V_m$ (V)')

# loop for step_end steps
for step in range(step_end):

  # Compute value of t
  t = step * dt

  # Get random number in correct range of -1 to 1 (will need to adjust output of np.random.random)
  random_num = 1 - 2*np.random.random()


  # Compute value of i at this time step
  i = i_mean * (1 + 0.1 * np.sqrt(t_max/dt)*random_num)

  # Compute v
  v = v + dt/tau * (el - v + r*i)

  # Plot v (using 'k.' to get even smaller markers)
  plt.plot(t, v, 'k.')


# Display plot
plt.show()










