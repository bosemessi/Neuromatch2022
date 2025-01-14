import numpy as np
import matplotlib.pyplot as plt


def plot_all(t_range, v, n, raster=None, spikes=None, spikes_mean=None):
    """
    Plots Time evolution for
    (1) multiple realizations of membrane potential
    (2) spikes
    (3) mean spike rate (optional)

    Args:
        t_range (numpy array of floats)
            range of time steps for the plots of shape (time steps)

        v (numpy array of floats)
            membrane potential values of shape (neurons, time steps)

        raster (numpy array of floats)
            spike raster of shape (neurons, time steps)

        spikes (dictionary of lists)
            list with spike times indexed by neuron number

        spikes_mean (numpy array of floats)
            Mean spike rate for spikes as dictionary

    Returns:
        Nothing.
    """

    v_mean = np.mean(v, axis=0)
    fig_w, fig_h = plt.rcParams['figure.figsize']
    plt.figure(figsize=(fig_w, 1.5 * fig_h))

    ax1 = plt.subplot(3, 1, 1)
    for j in range(n):
        plt.plot(t_range, v[j], color="k", alpha=0.5)
    # plt.plot(t_range, v_mean, 'C1', alpha=0.8, linewidth=3)
    plt.xticks([])
    plt.ylabel(r'$V_m$ (V)')

    if raster is not None:
        plt.subplot(3, 1, 2, sharex = ax1)
        spikes_mean = np.mean(raster, axis=0)
        plt.imshow(raster, cmap='Greys', origin='lower', aspect='auto', extent=(0, 0.15, -1, 1))

    else:
        plt.subplot(3, 1, 2, sharex=ax1)
        for j in range(n):
            times = np.array(spikes[j])
            plt.scatter(times, j * np.ones_like(times), color="C0", marker=".", alpha=0.2)

    plt.xticks([])
    plt.ylabel('neuron')

    if spikes_mean is not None:
        plt.subplot(3, 1, 3, sharex=ax1)
        plt.plot(t_range, spikes_mean)
        plt.xlabel('time (s)')
        plt.ylabel('rate (Hz)')

    plt.tight_layout()
    plt.show()