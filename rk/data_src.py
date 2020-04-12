from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def overlaid_histogram(data1, data2, data3, n_bins = 0, data1_name="", data1_color="#539caf", 
                       data2_name="", data2_color="#7663b0", data3_name = "", data3_color="#ffff00", 
                       x_label="", y_label="", title=""):
    # Set the bounds for the bins so that the two distributions are fairly compared
    max_nbins = 10
    data_range = [min(min(data1), min(data2), min(data3)), max(max(data1), max(data2), max(data3))]
    binwidth = (data_range[1] - data_range[0]) / max_nbins


    bins = n_bins

    # Create the plot
    _, ax = plt.subplots()
    ax.hist(data1, bins = bins, color = data1_color, alpha = 0.75, label = data1_name)
    ax.hist(data2, bins = bins, color = data2_color, alpha = 0.75, label = data2_name)
    ax.hist(data3, bins = bins, color = data3_color, alpha = 0.75, label = data3_name)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend(loc = 'best')

def ra_dec_distance(ra, dec, ra1, dec1):
    import numpy as np
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    c1 = SkyCoord(ra=ra, dec=dec, unit='deg')
    c2 = SkyCoord(ra=ra1, dec=dec1, unit='deg')
    sep = c1.separation(c2)
    return sep.degree

def histogram(data, n_bins = 0):
    import matplotlib.pyplot as plt
    max_nbins = 10
    data_range = [min(data), max(data)]
    bins = n_bins

    _, ax = plt.subplots()
    ax.hist(data, bins = bins, alpha = 1, label = "")
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_title("")
    ax.legend(loc = 'best')


