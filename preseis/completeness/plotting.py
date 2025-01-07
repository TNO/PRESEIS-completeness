from matplotlib import pyplot as plt
from .completeness import *


def plot_GMPE(epicentral_distance, local_magnitudes):
    """
    Reproduce Figure 2.7 from KNMI Report
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    def axes_settings(axes):
        axes.set_xlabel("Epicentral Distance [km]")
        axes.set_xlim([0, 100])
        axes.yaxis.grid(True, which="minor", alpha=0.5)
        axes.set_yscale("log")
        axes.set_ylabel("PGV [mm/s]")
        axes.set_ylim([10e-6, 10e1])

    # Plot for the Surface level:
    event_depth = 3.0
    station_depth = 0.0
    axes_settings(ax1)

    ax1.set_title("Ground Motion Prediction Equation Surface Geophones")
    colors = plt.cm.jet(np.linspace(0, 1, 8))
    for i, mag in enumerate(local_magnitudes):
        Y_mod_store = []
        for dist in epicentral_distance:
            Y_mod_store.append(
                ground_motion_model(mag, event_depth, station_depth, dist)
            )
        ax1.plot(epicentral_distance, Y_mod_store, label="M=" + str(mag), c=colors[i])
    ax1.legend()

    # Plot for the Deep level:
    event_depth = 3.0
    station_depth = 0.21
    axes_settings(ax2)

    ax2.set_title("Ground Motion Prediction Equation Geophones at 200m")
    colors = plt.cm.jet(np.linspace(0, 1, 8))
    for i, mag in enumerate(local_magnitudes):
        Y_mod_store = []
        for dist in epicentral_distance:
            Y_mod_store.append(
                ground_motion_model(mag, event_depth, station_depth, dist)
            )
        ax2.plot(epicentral_distance, Y_mod_store, label="M=" + str(mag), c=colors[i])
    ax2.legend()
    plt.show()


def plot_grid_cross_sections(grid, title, cbar_label, stationlocs=None):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5, wspace=0.8)
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]

    # find the middle indexes for x and y
    midx_index = int(len(grid["x"].data) / 2)
    midy_index = int(len(grid["y"].data) / 2)

    # find min and max values for colorbar:
    vmin = np.min(grid.data)
    vmax = np.max(grid.data)

    # z = 0m
    ax1.set_aspect("equal", "box")
    grid.isel(z=-1).plot(ax=ax1, x="x", y="y", vmin=vmin, vmax=vmax, add_colorbar=False)
    ax1.set_title("Z coord: " + str(grid["z"].data[-1]) + " [m]")
    ax1.axhline(grid["y"].data[midy_index], color="tab:cyan", ls="--")
    ax1.axvline(grid["x"].data[midx_index], color="tab:pink", ls="--")

    # z = base of model
    ax2.set_aspect("equal", "box")
    grid.isel(z=0).plot(ax=ax2, x="x", y="y", vmin=vmin, vmax=vmax, add_colorbar=False)
    ax2.set_title("Z coord: " + str(grid["z"].data[0]) + " [m]")
    ax2.axhline(grid["y"].data[midy_index], color="tab:cyan", ls="--")
    ax2.axvline(grid["x"].data[midx_index], color="tab:pink", ls="--")

    # x = middle of model
    grid.isel(x=midx_index).plot(
        ax=ax3, x="y", y="z", vmin=vmin, vmax=vmax, add_colorbar=False
    )
    ax3.set_title(
        "X coord: " + str(grid["x"].data[midx_index]) + " [m]", color="tab:pink"
    )

    # y = middle of model
    grid.isel(y=midy_index).plot(
        ax=ax4, x="x", y="z", vmin=vmin, vmax=vmax, cbar_kwargs={"label": cbar_label}
    )
    ax4.set_title(
        "Y coord: " + str(grid["y"].data[midy_index]) + " [m]", color="tab:cyan"
    )

    if stationlocs != None:
        plot_station_locations(ax1, stationlocs)
        plot_station_locations(ax2, stationlocs)
        plot_station_locations(ax3, stationlocs, vert_x=True)
        plot_station_locations(ax4, stationlocs, vert_y=True)

    plt.suptitle(title)
    plt.show()


def plot_station_locations(ax, stationlocs, vert_x=False, vert_y=False):
    # Plot station locations
    single_station = False
    if vert_x:
        if not single_station:
            for i in range(len(stationlocs["station"])):
                station_loc = stationlocs.isel(station=i)
                ax.scatter(
                    station_loc["y"], station_loc["z"], marker="v", color="red", s=100
                )
            else:
                ax.scatter(
                    stationlocs["y"], stationlocs["z"], marker="v", color="red", s=100
                )
        return

    if vert_y:
        if not single_station:
            for i in range(len(stationlocs["station"])):
                station_loc = stationlocs.isel(station=i)
                ax.scatter(
                    station_loc["x"], station_loc["z"], marker="v", color="red", s=100
                )
            else:
                ax.scatter(
                    stationlocs["x"], stationlocs["z"], marker="v", color="red", s=100
                )
        return

    for i in range(len(stationlocs["station"])):
        if not single_station:
            station_loc = stationlocs.isel(station=i)
            ax.scatter(
                station_loc["x"], station_loc["y"], marker="v", color="red", s=100
            )
        else:
            ax.scatter(
                stationlocs["x"], stationlocs["y"], marker="v", color="red", s=100
            )


def plot_magnitude_interpolation(combined_probability):
    combined_probability.plot(x="magnitude", y="z")
    plt.show()

    surface = combined_probability.isel(z=-1)
    surface.plot(x="magnitude")
    plt.ylabel("Probability")
    plt.show()


def plot_probabilities(pgm, pgm_sd, station_noise_mean, station_noise_sd, SNR):
    fig, ax = plt.subplots(1, 1)

    PGV_pdf = stats.norm(loc=pgm, scale=pgm_sd)
    noise_pdf = stats.norm(loc=station_noise_mean, scale=station_noise_sd)

    x = np.linspace(*PGV_pdf.interval(0.999))
    x2 = np.linspace(*noise_pdf.interval(0.999))
    ax.plot(x, PGV_pdf.pdf(x), label="PGV EQ")
    ax.plot(x2, noise_pdf.pdf(x2), label="Noise")

    ax.legend()
    ax.set_xlabel("ln(mm/s)")
    plt.show()
    plt.close()

    # now plot their comparison
    fig, ax = plt.subplots(1, 1)

    Z_mu = pgm - station_noise_mean
    Z_sigma = np.sqrt(pgm_sd**2 + station_noise_sd**2)
    Z = stats.norm(loc=Z_mu, scale=Z_sigma)
    x = np.linspace(*Z.interval(0.999))
    ax.plot(x, Z.pdf(x), label="pdf N(PGV - Noise)")
    ax.plot(x, 1 - Z.cdf(x), label="cdf N(PGV - Noise)")
    ax.axvline(SNR)
    ax.legend()
    ax.set_xlabel("ln(mm/s)")
    plt.show()
    plt.close()
