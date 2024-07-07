import io
import requests 
import numpy as np
import matplotlib.pyplot as plt

def load_spike_times_data(filename):
    r = requests.get(filename)
    if r.status_code != 200:
        print('Failed to download data')
    else:
        spike_times = np.load(io.BytesIO(r.content), allow_pickle=True)['spike_times']
    return spike_times

def plot_isis(single_neuron_isis, bins=50, histrange = None):
    if histrange:
        plt.hist(single_neuron_isis, bins=bins, histtype="stepfilled", range=histrange)
    else:
       plt.hist(single_neuron_isis, bins=bins, histtype="stepfilled")
    plt.axvline(single_neuron_isis.mean(), color="orange", label="Mean ISI")
    plt.xlabel("ISI duration (s)")
    plt.ylabel("Number of spikes")
    plt.legend()

def restrict_spike_times(spike_times, interval):
  """Given a spike_time dataset, restrict to spikes within given interval.

  Args:
    spike_times (sequence of np.ndarray): List or array of arrays,
      each inner array has spike times for a single neuron.
    interval (tuple): Min, max time values; keep min <= t < max.

  Returns:
    np.ndarray: like `spike_times`, but only within `interval`
  """
  interval_spike_times = []
  for spikes in spike_times:
    interval_mask = (spikes >= interval[0]) & (spikes < interval[1])
    interval_spike_times.append(spikes[interval_mask])
  return np.array(interval_spike_times, object)

def exponential(xs, scale, rate, x0):
  """A simple parameterized exponential function, applied element-wise.

  Args:
    xs (np.ndarray or float): Input(s) to the function.
    scale (float): Linear scaling factor.
    rate (float): Exponential growth (positive) or decay (negative) rate.
    x0 (float): Horizontal offset.

  """
  ys = scale * np.exp(rate * (xs - x0))
  return ys

def inverse(xs, scale, x0):
  """A simple parameterized inverse function (`1/x`), applied element-wise.

  Args:
    xs (np.ndarray or float): Input(s) to the function.
    scale (float): Linear scaling factor.
    x0 (float): Horizontal offset.

  """
  ys = scale / (xs - x0)
  return ys

def linear(xs, slope, y0):
  """A simple linear function, applied element-wise.

  Args:
    xs (np.ndarray or float): Input(s) to the function.
    slope (float): Slope of the line.
    y0 (float): y-intercept of the line.

  """
  ys = slope * xs + y0
  return ys

