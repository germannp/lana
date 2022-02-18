"""Analyze and plot cell motility from tracks"""
from collections import OrderedDict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import ConvexHull
from statsmodels.stats.proportion import proportion_confint

from lana.utils import equalize_axis3d
from lana.utils import track_identifiers


def _uniquize_tracks(tracks, verbose=False):
    """Cluster tracks, if not unique"""
    if "Time" not in tracks.columns:
        return

    tracks["Orig. Index"] = tracks.index
    if not tracks.index.is_unique:
        tracks.reset_index(drop=True, inplace=True)

    if "Track_ID" in tracks.columns:
        max_track_id = tracks["Track_ID"].max()
    else:
        max_track_id = 0

    for identifiers, track in tracks.groupby(track_identifiers(tracks)):
        if sum(track["Time"].duplicated()) != 0:
            n_clusters = track["Time"].value_counts().max()
            track = track.copy()
            index = track.index
            if "Track_ID" in track.columns:
                tracks.loc[index, "Orig. Track_ID"] = track["Track_ID"]

            clusters = AgglomerativeClustering(n_clusters).fit(track[["X", "Y", "Z"]])
            track.loc[:, "Cluster"] = clusters.labels_

            if sum(track[["Cluster", "Time"]].duplicated()) != 0:
                clusters = AgglomerativeClustering(n_clusters).fit(
                    track[["Orig. Index"]]
                )
                track.loc[:, "Cluster"] = clusters.labels_

            if sum(track[["Cluster", "Time"]].duplicated()) == 0:
                tracks.loc[index, "Track_ID"] = max_track_id + 1 + clusters.labels_
                max_track_id += n_clusters
                pd.set_option("display.max_rows", 1000)
                if verbose:
                    print(
                        f"  Warning: Split non-unique track {identifiers} by clustering."
                    )
            else:
                tracks.drop(index, inplace=True)
                if verbose:
                    print(f"  Warning: Delete non-unique track {identifiers}.")


def _split(tracks, skip, warning=None):
    """Split a track where skip(track) > 0"""
    if "Track_ID" in tracks.columns:
        max_track_id = tracks["Track_ID"].max()
    else:
        max_track_id = 0

    for criterium, track in tracks.groupby(track_identifiers(tracks)):
        skips = skip(track)
        if skips.max() > 0:
            index = track.index
            if "Track_ID" in track.columns:
                tracks.loc[index, "Orig. Track_ID"] = track["Track_ID"]
            skip_sum = skips.cumsum()
            tracks.loc[index, "Track_ID"] = max_track_id + 1 + skip_sum
            max_track_id += max(skip_sum) + 1
            if warning:
                print(warning.format(criterium))


def _split_at_skip(tracks, jump_threshold=None, verbose=False):
    """Split track if timestep is missing or a step too long"""
    if "Time" not in tracks.columns:
        print("  Warning: No times given.")
        return

    if not tracks.index.is_unique:
        tracks.reset_index(drop=True, inplace=True)
        print("  Warning: Non-unique index, resetting it.")

    def non_uniform_timestep(track):
        timesteps = track["Time"].diff()
        return ((timesteps - timesteps.min()) / timesteps.min()).round().fillna(0)

    _split(
        tracks,
        non_uniform_timestep,
        "Split track {} with non-uniform timesteps." * verbose,
    )

    if jump_threshold is None:
        return

    def jump(track):
        positions = track[["X", "Y", "Z"]]
        dr = positions.diff()
        dr_norms = np.linalg.norm(dr, axis=1)
        return np.nan_to_num(dr_norms) > jump_threshold

    _split(
        tracks,
        jump,
        "Split track {} with jump > {}um.".format({}, jump_threshold) * verbose,
    )


def analyze(
    raw_tracks,
    uniform_timesteps=True,
    min_length=6,
    jump_threshold=None,
    repeat_portion=0.25,
    verbose=False,
):
    """Return dataframe with velocity, turning angle & plane angle"""
    print("\nAnalyzing tracks")

    tracks = raw_tracks.copy()

    if "Time" not in tracks.columns:
        print("  Warning: No time given, using index!")
        tracks["Time"] = tracks.index
        if not tracks.index.is_unique:  # For inplace analysis!
            tracks.reset_index(drop=True, inplace=True)
    else:
        # TODO: Test for unique, but unsorted times.
        tracks = tracks.sort_values(track_identifiers(tracks) + ["Time"])
        _uniquize_tracks(tracks, verbose)
        if uniform_timesteps:
            _split_at_skip(tracks, jump_threshold, verbose)

    if not verbose and "Orig. Track_ID" in tracks.columns:
        print("  Some tracks were split, verbose=True for more info.")

    n_i = tracks.Track_ID.unique().size
    for criterium, track in tracks.groupby(track_identifiers(tracks)):
        track_length = len(track)
        if track_length < min_length:
            tracks.drop(track.index, inplace=True)
            if verbose:
                print(f"  Delete track {criterium} with {len(track)} timesteps.")
            continue

        # Can't `continue` outer loop in Python, so no loop over coordinates
        # TODO: Could be tested as well.
        if (repeats := track["X"].duplicated().sum()) > track_length * repeat_portion:
            tracks.drop(track.index, inplace=True)
            if verbose:
                print(f"  Delete track {criterium} with {repeats} X repeats.")
            continue

        if (repeats := track["Y"].duplicated().sum()) > track_length * repeat_portion:
            tracks.drop(track.index, inplace=True)
            if verbose:
                print(f"  Delete track {criterium} with {repeats} Y repeats.")
            continue

        if (
            "Z" in track.columns
            and (repeats := track["Z"].duplicated().sum())
            > track_length * repeat_portion
        ):
            tracks.drop(track.index, inplace=True)
            if verbose:
                print(f"  Delete track {criterium} with {repeats} Z repeats.")
            continue

        tracks.loc[track.index, "Track Time"] = (
            track["Time"] - track["Time"].iloc[0]
        ).round(4)

        if "Z" in track.columns:
            positions = track[["X", "Y", "Z"]]
        else:
            positions = track[["X", "Y"]].copy()
            positions["Z"] = 0

        tracks.loc[track.index, "Displacement"] = np.linalg.norm(
            positions - positions.iloc[0], axis=1
        )

        dr = positions.diff()
        dr_norms = np.linalg.norm(dr, axis=1)

        tracks.loc[track.index, "Velocity"] = dr_norms / track["Time"].diff()
        assert all(
            tracks.loc[track.index, "Velocity"].dropna() >= 0
        ), f"Track {criterium} has negative velocity."

        dot_products = np.sum(dr.shift(-1) * dr, axis=1)
        norm_products = dr_norms[1:] * dr_norms[:-1]

        tracks.loc[track.index, "Turning Angle"] = np.arccos(
            dot_products[:-1] / norm_products
        )

        tracks.loc[track.index, "Plane Angle"] = np.nan

        n_vectors = np.cross(dr, dr.shift())
        n_norms = np.linalg.norm(n_vectors, axis=1)
        dot_products = np.sum(n_vectors[1:] * n_vectors[:-1], axis=1)
        norm_products = n_norms[1:] * n_norms[:-1]
        angles = np.arccos(dot_products / norm_products)
        cross_products = np.cross(n_vectors[1:], n_vectors[:-1])
        cross_dot_dr = np.sum(cross_products[2:] * dr.values[2:-1], axis=1)
        cross_norms = np.linalg.norm(cross_products[2:], axis=1)
        signs = cross_dot_dr / cross_norms / dr_norms[2:-1]

        if "Z" in track.columns:
            tracks.loc[track.index[2:-1], "Plane Angle"] = signs * angles[2:]
        else:
            tracks.loc[track.index[2:-1], "Plane Angle"] = angles[2:]

    n_f = tracks.Track_ID.unique().size
    if not verbose and n_f != n_i:
        print("  Warning: Some tracks were deleted, verbose=True for more info.")

    return tracks


def plot_tracks(
    raw_tracks,
    summary=None,
    draw_turns=True,
    n_tracks=25,
    condition="Condition",
    context="notebook",
    save=False,
):
    """Plot tracks"""
    tracks = raw_tracks.copy()
    _uniquize_tracks(tracks)
    _split_at_skip(tracks)

    def condition_changes(track):
        changes = np.diff(track[condition].factorize()[0])
        return np.hstack((0, changes))

    n_tracks_before = len(tracks["Track_ID"].unique())
    _split(tracks, condition_changes, "")
    if len(tracks["Track_ID"].unique()) != n_tracks_before:
        print(f"  Warning: Split tracks with several {condition}")

    if type(summary) == pd.core.frame.DataFrame:
        skip_steps = int(
            next(
                word
                for column in summary.columns
                for word in column.split()
                if word.isdigit()
            )
        )

    if summary is not None and draw_turns:
        alpha = 0.33
    else:
        alpha = 1

    if condition not in tracks.columns:
        tracks[condition] = "Default"
    n_conditions = len(tracks[condition].unique())

    sns.set(style="ticks", context=context)
    fig = plt.figure(figsize=(12, 12))
    if "Z" in tracks.columns:
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111, aspect="equal")
    labels = []
    for i, (cond, cond_tracks) in enumerate(tracks.groupby(condition)):
        if summary is not None and draw_turns:
            cond_summary = summary[summary[condition] == cond]
            max_turn_column = next(
                column for column in summary.columns if column.startswith("Max. Turn")
            )
            if len(cond_tracks["Track_ID"].unique()) > n_tracks / n_conditions:
                choice = cond_summary.sort_values(max_turn_column, ascending=False)[
                    "Track_ID"
                ][: int(n_tracks / n_conditions)]
                cond_tracks = cond_tracks[cond_tracks["Track_ID"].isin(choice)]
        elif len(cond_tracks["Track_ID"].unique()) > n_tracks / n_conditions:
            choice = np.random.choice(
                cond_tracks["Track_ID"].unique(),
                int(n_tracks / n_conditions),
                replace=False,
            )
            cond_tracks = cond_tracks[cond_tracks["Track_ID"].isin(choice)]

        color = sns.color_palette(n_colors=i + 1)[-1]
        for j, (_, track) in enumerate(
            cond_tracks.groupby(track_identifiers(cond_tracks))
        ):
            labels.append(cond)
            track_id = track["Track_ID"].iloc[0]
            if "Z" in tracks.columns:
                ax.plot(
                    track["X"].values,
                    track["Y"].values,
                    track["Z"].values,
                    color=color,
                    alpha=alpha,
                    label=track_id,
                    pickradius=5,
                )
            else:
                ax.plot(
                    track["X"].values,
                    track["Y"].values,
                    color=color,
                    alpha=alpha,
                    label=track_id,
                    pickradius=5,
                )
            if summary is not None and draw_turns:
                turn_time = cond_summary[cond_summary["Track_ID"] == track_id][
                    "Turn Time"
                ]
                turn_loc = track.index.get_loc(
                    track[np.isclose(track["Time"], turn_time.values[0])].index.values[
                        0
                    ]
                )
                turn_times = track["Time"][turn_loc - 1 : turn_loc + skip_steps]
                turn = track[track["Time"].isin(turn_times)]
                if "Z" in tracks.columns:
                    ax.plot(
                        turn["X"].values,
                        turn["Y"].values,
                        turn["Z"].values,
                        color=color,
                    )
                else:
                    ax.plot(turn["X"].values, turn["Y"].values, color=color)

    def on_pick(event):
        track_id = event.artist.get_label()
        if summary is not None:
            print(
                summary[summary["Track_ID"] == float(track_id)][
                    ["Track_ID", "Condition", "Mean Velocity", "Track Duration"]
                ]
            )
        else:
            print("Track_ID: " + track_id)

    fig.canvas.mpl_connect("pick_event", on_pick)

    if "Z" in tracks.columns:
        equalize_axis3d(ax)
    else:
        sns.despine()
    handles, _ = ax.get_legend_handles_labels()
    unique_entries = OrderedDict(zip(labels, handles))
    ax.legend(unique_entries.values(), unique_entries.keys(), frameon=False)
    plt.tight_layout()

    if save:
        conditions = [cond.replace("= ", "") for cond in tracks[condition].unique()]
        plt.savefig("Tracks" + "-".join(conditions) + ".png", dpi=300)
    else:
        plt.show()


def plot(
    tracks,
    save=False,
    palette="deep",
    max_time=9,
    condition="Condition",
    plot_each_sample=False,
    context="notebook",
    plot_plane_angle=True,
):
    """Plot aspects of motility for different conditions"""
    if "Displacement" not in tracks.columns:
        tracks = analyze(tracks)

    if condition not in tracks.columns:
        tracks[condition] = "Default"

    sns.set(
        style="ticks",
        palette=sns.color_palette(palette, len(tracks[condition].unique())),
        context=context,
    )
    if "Plane Angle" in tracks.columns and plot_plane_angle:
        figure, axes = plt.subplots(ncols=4, figsize=(16, 5.5))
    else:
        figure, axes = plt.subplots(ncols=3, figsize=(12, 5.5))
    plt.setp(axes, yticks=[])
    plt.setp(axes, xticks=[])

    axes[0].set_ylabel("Median Displacement")
    axes[0].set_xlabel(r"Sqrt. of Time [min$^{1/2}$]")
    axes[0].set_xlim([0, np.sqrt(max_time)])
    axes[0].set_xticks([0, np.sqrt(max_time)])
    axes[0].set_xticklabels(["0", str(int(np.sqrt(max_time)))])

    axes[1].set_xlabel(r"Velocity  [$\mu$m/min]")
    axes[1].set_ylabel("Density")
    axes[1].set_xlim(0, np.percentile(tracks["Velocity"].dropna(), 99.5))
    axes[1].set_xticks([0, 10])
    axes[1].set_xticklabels(["0", "10"])

    axes[2].set_xlabel("Turning Angle")
    axes[2].set_ylabel("Density")
    axes[2].set_xlim([0, np.pi])
    axes[2].set_xticks([0, np.pi / 2, np.pi])
    axes[2].set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$"])

    if "Plane Angle" in tracks.columns and plot_plane_angle:
        axes[3].set_xlabel("Plane Angle")
        axes[3].set_ylabel("Density")
        axes[3].set_xlim([-np.pi, np.pi])
        axes[3].set_xticks([-np.pi, 0, np.pi])
        axes[3].set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])

    if plot_each_sample:
        groups = [condition, "Sample"]
    else:
        groups = condition

    for i, (_, cond_tracks) in enumerate(tracks.groupby(groups)):
        # Plot displacements, inspired by http://stackoverflow.com/questions/
        # 22795348/plotting-time-series-data-with-seaborn
        label = cond_tracks[condition].iloc[0]
        color = sns.color_palette()[i]
        displacements = (
            cond_tracks[["Track Time", "Displacement"]]
            .groupby("Track Time")
            .describe()["Displacement"]
        )
        if max_time:
            displacements = displacements[displacements.index <= max_time]
        axes[0].plot(
            np.sqrt(displacements.index), displacements["50%"], label=label, color=color
        )
        if not plot_each_sample:
            axes[0].fill_between(
                np.sqrt(displacements.index),
                displacements["25%"],
                displacements["75%"],
                alpha=0.2,
                color=color,
            )
        if not plot_each_sample and len(tracks[condition].unique()) == 1:
            axes[0].fill_between(
                np.sqrt(displacements.index),
                displacements["min"],
                displacements["max"],
                alpha=0.2,
                color=color,
            )

        # Plot velocities
        sns.kdeplot(
            cond_tracks["Velocity"].dropna(),
            clip=(0, np.inf),
            shade=not plot_each_sample,
            ax=axes[1],
            gridsize=500,
            label="",
            color=color,
        )

        # Plot turning angles
        turning_angles = cond_tracks["Turning Angle"].dropna().values
        if "Z" in tracks.columns:
            x = np.arange(0, np.pi, 0.1)
            axes[2].plot(x, np.sin(x) / 2, "--k")
            sns.kdeplot(
                turning_angles,
                clip=(0, np.inf),
                color=color,
                shade=not plot_each_sample,
                ax=axes[2],
            )
        else:
            turning_angles = np.concatenate(
                (  # Mirror at boundaries.
                    -turning_angles,
                    turning_angles,
                    2 * np.pi - turning_angles,
                )
            )
            axes[2].plot([0, np.pi], [1 / (3 * np.pi), 1 / (3 * np.pi)], "--k")
            sns.kdeplot(
                turning_angles, color=color, shade=not plot_each_sample, ax=axes[2]
            )

        # Plot Plane Angles
        if "Plane Angle" in tracks.columns and plot_plane_angle:
            plane_angles = cond_tracks["Plane Angle"].dropna().values
            plane_angles = np.concatenate(
                (  # Mirror at boundaries.
                    -2 * np.pi + plane_angles,
                    plane_angles,
                    2 * np.pi + plane_angles,
                )
            )
            axes[3].plot([-np.pi, np.pi], [1 / (6 * np.pi), 1 / (6 * np.pi)], "--k")
            sns.kdeplot(
                plane_angles, color=color, shade=not plot_each_sample, ax=axes[3]
            )

    handles, labels = axes[0].get_legend_handles_labels()
    unique_entries = OrderedDict(zip(labels, handles))
    axes[0].legend(
        unique_entries.values(), unique_entries.keys(), loc="upper left", frameon=False
    )

    sns.despine()
    plt.tight_layout()
    if save:
        conditions = [cond.replace("= ", "") for cond in tracks[condition].unique()]
        plt.savefig(
            "Motility_"
            + "-".join(conditions)
            + "_all-samples" * plot_each_sample
            + ".png",
            dpi=300,
        )
    else:
        plt.show()


def plot_dr(raw_tracks, save=False, condition="Condition", context="notebook"):
    """Plot the differences in X, Y (and Z) to show biases"""
    tracks = raw_tracks.copy()
    _uniquize_tracks(tracks)
    _split_at_skip(tracks)

    dimensions = [dim for dim in ["X", "Y", "Z"] if dim in tracks.columns]

    differences = pd.DataFrame()

    for _, track in tracks.groupby(track_identifiers(tracks)):
        differences = differences.append(track[dimensions].diff().dropna())
        if "Track_ID" in differences.columns:
            differences = differences.fillna(track["Track_ID"].iloc[0])
        else:
            differences["Track_ID"] = track["Track_ID"].iloc[0]

    sns.set(style="ticks", palette="deep", context=context)
    fig, axes = plt.subplots(ncols=3, figsize=(15.5, 5.5))
    plt.setp(axes, yticks=[])
    plt.setp(axes, xticks=[])

    axes[0].set_title(r"$\Delta \vec r$")
    axes[0].set_xticks([0])
    axes[0].set_xticklabels([r"$0$"])

    for dimension in dimensions:
        sns.kdeplot(differences[dimension], shade=True, ax=axes[0])

    axes[1].set_title("Joint Distribution")
    axes[1].set_xlabel(r"$\Delta x$")
    axes[1].set_ylabel(r"$\Delta y$")
    axes[1].axis("equal")
    axes[1].set_xlim([differences["X"].quantile(0.1), differences["X"].quantile(0.9)])
    axes[1].set_ylim([differences["Y"].quantile(0.1), differences["Y"].quantile(0.9)])
    sns.kdeplot(data=differences, x="X", y="Y", shade=False, cmap="Greys", ax=axes[1])

    axes[2].set_title(r"$\Delta \vec r$ Lag Plot")
    axes[2].axis("equal")
    axes[2].set_xlabel(r"$\Delta r_i(t)$")
    axes[2].set_ylabel(r"$\Delta r_i(t+1)$")
    for i, dim in enumerate(dimensions):
        color = sns.color_palette()[i]
        for _, track in differences.groupby("Track_ID"):
            axes[2].scatter(track[dim], track[dim].shift(), facecolors=color)

    sns.despine()
    plt.tight_layout()
    if save:
        conditions = [cond.replace("= ", "") for cond in tracks[condition].unique()]
        plt.savefig("dr_" + "-".join(conditions) + ".png", dpi=300)
    else:
        plt.show()


def joint_plot(
    tracks,
    condition="Condition",
    save=False,
    palette="deep",
    skip_color=0,
    context="notebook",
):
    """Plot the joint distribution of the velocities and turning angles."""
    if "Displacement" not in tracks.columns:
        tracks = analyze(tracks)

    if condition not in tracks.columns:
        tracks[condition] = "Default"

    sns.set(
        style="white",
        palette=sns.color_palette(
            palette, tracks[condition].unique().__len__() + skip_color
        ),
        context=context,
    )

    y_upper_lim = np.percentile(tracks["Velocity"].dropna(), 99.5)

    for i, (cond, cond_tracks) in enumerate(tracks.groupby(condition)):
        color = sns.color_palette()[i + skip_color]
        sns.jointplot(
            data=cond_tracks,
            x="Turning Angle",
            y="Velocity",
            kind="kde",
            xlim=[0, np.pi],
            space=0,
            color=color,
            ylim=[0, y_upper_lim],
            joint_kws={"shade": False},
        )
        if save:
            plt.savefig("Joint-Motility_" + cond.replace("= ", "") + ".png", dpi=300)
        else:
            plt.show()


def plot_tracks_parameter_space(
    tracks,
    n_tracks=None,
    condition="Condition",
    save=False,
    palette="deep",
    skip_color=0,
    context="notebook",
):
    """Plot tracks in velocities-turning-angles-space"""
    if "Displacement" not in tracks.columns:
        tracks = analyze(tracks)

    if condition not in tracks.columns:
        tracks[condition] = "Default"

    sns.set(
        style="ticks",
        palette=sns.color_palette(
            palette, tracks[condition].unique().__len__() + skip_color
        ),
        context=context,
    )
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_xlabel("Turning Angle")
    ax.set_xlim([0, np.pi])
    ax.set_xticks([0, np.pi / 2, np.pi])
    ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$"])
    ax.set_ylabel("Velocity")
    for i, (_, cond_tracks) in enumerate(tracks.groupby(condition)):
        color = sns.color_palette()[i + skip_color]
        if n_tracks != None:
            cond_tracks = cond_tracks[
                cond_tracks["Track_ID"].isin(
                    np.random.choice(cond_tracks["Track_ID"], n_tracks)
                )
            ]
        for _, track in cond_tracks.groupby("Track_ID"):
            ax.plot(track["Turning Angle"], track["Velocity"], color=color, alpha=0.5)

    sns.despine()
    plt.tight_layout()
    if save:
        conditions = [cond.replace("= ", "") for cond in tracks[condition].unique()]
        plt.savefig(
            "Motility-TracksInParameterSpace_" + "-".join(conditions) + ".png", dpi=300
        )
    else:
        plt.show()


def plot_arrest(
    tracks, condition="Condition", arrest_velocity=3, save=False, context="notebook"
):
    """Plot velocity aligned to minimum and distribution of arrested steps"""
    if "Displacement" not in tracks.columns:
        tracks = analyze(tracks)

    if condition not in tracks.columns:
        tracks[condition] = "Default"

    sns.set(style="ticks", context=context)
    fig, axes = plt.subplots(1, 2, figsize=(8, 5.5))
    axes[0].set_xlabel("Time to minimum")
    axes[0].set_ylabel("Velocity")
    axes[1].set_xlabel(r"Consecutive steps below {} $\mu$m/min".format(arrest_velocity))
    axes[1].set_ylabel("Proportion")

    for i, (cond, cond_tracks) in enumerate(tracks.groupby(condition)):
        velocities = pd.Series(dtype="float64")
        arrested_segment_lengths = []
        for _, track in cond_tracks.groupby(track_identifiers(cond_tracks)):
            min_index = track["Velocity"].idxmin()
            track_velocities = pd.Series(
                track["Velocity"].values, track["Time"] - track.loc[min_index, "Time"]
            )
            velocities = velocities.append(track_velocities.dropna())
            arrested = track["Velocity"] < arrest_velocity
            arrested_segments = np.split(arrested, np.where(np.diff(arrested))[0] + 1)
            arrested_segment_lengths.extend(
                [sum(segment) for segment in arrested_segments if sum(segment) > 0]
            )

        velocities.index = np.round(velocities.index, 5)  # Handle non-integer 'Times'
        arrestats = velocities.groupby(velocities.index).describe()

        color = sns.color_palette(n_colors=i + 1)[-1]
        axes[0].plot(arrestats.index, arrestats["50%"], color=color)
        axes[0].fill_between(
            arrestats.index, arrestats["25%"], arrestats["75%"], color=color, alpha=0.2
        )
        axes[0].fill_between(
            arrestats.index, arrestats["min"], arrestats["max"], color=color, alpha=0.2
        )

        sns.histplot(
            arrested_segment_lengths,
            bins=np.arange(1, max(arrested_segment_lengths) + 1) - 0.5,
            stat="density",
            kde=False,
            color=color,
            ax=axes[1],
        )

    axes[0].set_xlim([-3, 3])
    axes[1].get_xaxis().set_major_locator(plt.MaxNLocator(integer=True))
    sns.despine()
    plt.tight_layout()
    if save:
        conditions = [cond.replace("= ", "") for cond in tracks[condition].unique()]
        plt.savefig("Arrest_" + "-".join(conditions) + ".png", dpi=300)
    else:
        plt.show()


def lag_plot(
    tracks,
    condition="Condition",
    save=False,
    palette="deep",
    skip_color=0,
    null_model=True,
    context="notebook",
):
    """Lag plot for velocities and turning angles"""
    if "Displacement" not in tracks.columns:
        tracks = analyze(tracks)

    if condition not in tracks.columns:
        tracks[condition] = "Default"

    sns.set(
        style="white",
        palette=sns.color_palette(
            palette, tracks[condition].unique().__len__() + skip_color
        ),
        context=context,
    )
    if "Plane Angle" in tracks.columns:
        fig, ax = plt.subplots(1, 3, figsize=(12, 4.25))
    else:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4.25))
    plt.setp(ax, yticks=[])
    plt.setp(ax, xticks=[])
    ax[0].set_title("Velocity")
    ax[0].set_xlabel("v(t)")
    ax[0].set_ylabel("v(t+1)")
    ax[0].axis("equal")
    ax[1].set_title("Turning Angle")
    ax[1].set_xlabel(r"$\theta$(t)")
    ax[1].set_ylabel(r"$\theta$(t+1)")
    ax[1].axis("equal")
    if "Plane Angle" in tracks.columns:
        ax[2].set_title("Plane Angle")
        ax[2].set_xlabel(r"$\phi$(t)")
        ax[2].set_ylabel(r"$\phi$(t+1)")
        ax[2].axis("equal")

    if null_model:
        null_model = tracks.sample(len(tracks))
        ax[0].scatter(
            null_model["Velocity"], null_model["Velocity"].shift(), facecolors="0.8"
        )
        ax[1].scatter(
            null_model["Turning Angle"],
            null_model["Turning Angle"].shift(),
            facecolors="0.8",
        )
        if "Plane Angle" in tracks.columns:
            ax[2].scatter(
                null_model["Plane Angle"],
                null_model["Plane Angle"].shift(),
                facecolors="0.8",
            )

    for i, (_, cond_tracks) in enumerate(tracks.groupby(condition)):
        color = sns.color_palette()[i + skip_color]
        for _, track in cond_tracks.groupby("Track_ID"):
            ax[0].scatter(
                track["Velocity"], track["Velocity"].shift(), facecolors=color
            )
            ax[1].scatter(
                track["Turning Angle"], track["Turning Angle"].shift(), facecolors=color
            )
            if "Plane Angle" in tracks.columns:
                ax[2].scatter(
                    track["Plane Angle"], track["Plane Angle"].shift(), facecolors=color
                )

    sns.despine()
    plt.tight_layout()
    if save:
        conditions = [cond.replace("= ", "") for cond in tracks[condition].unique()]
        plt.savefig("Motility-LagPlot_" + "-".join(conditions) + ".png", dpi=300)
    else:
        plt.show()


def summarize(tracks, arrest_velocity=3, skip_steps=4, verbose=False):
    """Summarize track statistics, e.g. mean velocity per track"""
    if "Displacement" not in tracks.columns:
        tracks = analyze(tracks)

    print("\nSummarizing track statistics")

    summary = pd.DataFrame()

    identifiers = track_identifiers(tracks)
    for i, (_, track) in enumerate(tracks.groupby(identifiers)):
        if verbose:
            print(
                f"  Analysing "
                + ", ".join(
                    [str(track.iloc[0][identifier]) for identifier in identifiers]
                )
            )
        if "Track_ID" in track.columns:
            track_id = track.iloc[0]["Track_ID"]
            summary.loc[i, "Track_ID"] = track_id
        if "Condition" in track.columns:
            summary.loc[i, "Condition"] = track.iloc[0]["Condition"]
        else:
            summary.loc[i, "Condition"] = "Default"
        if "Sample" in track.columns:
            summary.loc[i, "Sample"] = track.iloc[0]["Sample"]
        if "Tissue" in track.columns:
            summary.loc[i, "Tissue"] = track.iloc[0]["Tissue"]

        summary.loc[i, "Mean Velocity"] = track["Velocity"].mean()
        summary.loc[i, "Mean Turning Angle"] = track["Turning Angle"].mean()
        if "Plane Angle" in track.columns:
            summary.loc[i, "Mean Plane Angle"] = track["Plane Angle"].mean()

        summary.loc[i, "Track Duration"] = (
            track["Time"].iloc[-1] - track["Time"].iloc[0]
        )

        summary.loc[i, "Arrest Coefficient"] = len(
            track[track["Velocity"] < arrest_velocity]
        ) / len(track["Velocity"].dropna())

        if "Z" in track.columns:
            positions = track[["X", "Y", "Z"]]
            ndim = 3
        else:
            positions = track[["X", "Y"]]
            ndim = 2

        summary.loc[i, "Motility Coefficient"] = (
            np.pi
            * track["Displacement"].iloc[-1]
            / (2 * ndim)
            / track["Track Time"].max()
        )

        dr = positions.diff()
        dr_norms = np.linalg.norm(dr, axis=1)

        summary.loc[i, "Confinement Ratio"] = (
            track["Displacement"].iloc[-1] / dr_norms[1:].sum()
        )

        summary.loc[i, "Corr. Confinement Ratio"] = (
            track["Displacement"].iloc[-1]
            / dr_norms[1:].sum()
            * np.sqrt(track["Track Time"].max())
        )

        summary.loc[i, "Mean Sq. Velocity Lag"] = np.mean(track["Velocity"].diff() ** 2)

        summary.loc[i, "Mean Sq. Turn. Angle Lag"] = np.mean(
            track["Turning Angle"].diff() ** 2
        )

        if len(track) > skip_steps + 1:
            dot_products = np.sum(dr.shift(-skip_steps) * dr, axis=1)
            norm_products = dr_norms[skip_steps:] * dr_norms[:-skip_steps]
            turns = np.arccos(dot_products.iloc[1:-skip_steps] / norm_products[1:])

            summary.loc[i, f"Max. Turn Over {skip_steps + 1} Steps"] = max(turns)

            summary.loc[i, "Turn Time"] = track.loc[turns.idxmax(), "Time"]

            cross_product = np.cross(
                dr.shift(-skip_steps).loc[turns.idxmax()], dr.loc[turns.idxmax()]
            )
            normal_vec = cross_product / np.linalg.norm(cross_product)

            summary.loc[i, "Skew Lines Distance"] = abs(
                np.sum(
                    (
                        positions.shift(-skip_steps).loc[turns.idxmax()]
                        - positions.loc[turns.idxmax()]
                    )
                    * normal_vec
                )
            )

        hull = ConvexHull(positions)
        summary.loc[i, "Scan. Area/Step"] = hull.area / len(track)
        summary.loc[i, "Scan. Vol./Step"] = hull.volume / len(track)

        if "Surface Area (µm2)" in track.columns:
            summary.loc[i, "Mean Surface Area (µm2)"] = track[
                "Surface Area (µm2)"
            ].mean()

        if "Volume (µm3)" in track.columns:
            summary.loc[i, "Mean Volume (µm3)"] = track["Volume (µm3)"].mean()

        if "Surface Area (µm2)" in track.columns and "Volume (µm3)" in track.columns:
            summary.loc[i, "Mean Sphericity"] = (
                np.pi ** (1 / 3)
                * (6 * track["Volume (µm3)"]) ** (2 / 3)
                / track["Surface Area (µm2)"]
            ).mean()

    for cond, cond_summary in summary.groupby("Condition"):
        n_tracks = len(cond_summary)
        n_steps = len(tracks[tracks["Condition"] == cond])
        print(f"  {n_tracks} tracks in {cond} with {n_steps} timesteps in total.")

    return summary


def plot_summary(summary, save=False, condition="Condition", context="notebook"):
    """Plot distributions and joint distributions of the track summary"""
    to_drop = [
        column
        for column in summary.columns
        if (
            column not in [condition, "Sample"]
            and summary[column].var() == 0
            or "Turn " in column
        )
    ]
    to_drop.extend(
        [
            column
            for column in [
                "Track_ID",
                "Skew Lines Distance",
                "Mean Sq. Turn. Angle Lag",
                "Mean Sq. Velocity Lag",
                "Scan. Area/Step",
                "Scan. Vol./Step",
                "Mean Surface Area (µm2)",
                "Mean Volume (µm3)",
            ]
            if column in summary.columns
        ]
    )

    sns.set(style="white", context=context)
    sns.pairplot(summary.drop(to_drop, axis=1), hue=condition, diag_kind="kde")
    plt.tight_layout()

    if save:
        conditions = [cond.replace("= ", "") for cond in summary[condition].unique()]
        plt.savefig("Summary_" + "-".join(conditions) + ".png", dpi=300)
    else:
        plt.show()


def plot_uturns(
    summary,
    critical_rad=2.9,
    time_step=20,
    save=False,
    condition="Condition",
    context="notebook",
):
    """Plot and print steepest turns over more than critical_rad"""
    turn_column = next(
        col for col in summary.columns if col.startswith("Max. Turn Over")
    )
    columns_of_interest = [
        "Skew Lines Distance",
        "Mean Velocity",
        "Arrest Coefficient",
        condition,
        turn_column,
    ]

    big_turns = summary[summary[turn_column] > critical_rad]
    mean_steps = big_turns["Mean Velocity"] * time_step / 60
    uturns = big_turns[big_turns["Skew Lines Distance"] < mean_steps]

    skip_steps = int(next(word for word in turn_column.split() if word.isdigit()))
    print(
        f"\nPlotting turns with more than {critical_rad} rad over {skip_steps} steps narrower than a mean step"
    )
    for cond, cond_uturns in uturns.groupby(condition):
        n_tracks = len(summary[summary[condition] == cond])
        n_turns = len(cond_uturns)
        ci_low, ci_upp = proportion_confint(n_turns, n_tracks, method="wilson")
        print(
            "  {:5.2f}% [{:5.2f}, {:5.2f}] tracks in {} with U-Turns ({} of {}).".format(
                n_turns / n_tracks * 100,
                ci_low * 100,
                ci_upp * 100,
                cond,
                n_turns,
                n_tracks,
            )
        )

    print("Binomial proportion 95% CIs are Wilson Score intervals.")

    sns.set(style="white", context=context)
    sns.pairplot(uturns[columns_of_interest], hue=condition, diag_kind="kde")
    plt.tight_layout()

    if save:
        conditions = [cond.replace("= ", "") for cond in summary[condition].unique()]
        plt.savefig(
            "U-Turns_"
            + "-".join(conditions)
            + "_{:1.1f}over{}steps.png".format(critical_rad, skip_steps),
            dpi=300,
        )
    else:
        plt.show()


def plot_shapes(summary, save=False, condition="Condition", context="notebook"):
    """Plot and print area and volume of all steps and averaged over track"""
    columns_of_interest = [
        "Scan. Area/Step",
        "Scan. Vol./Step",
        "Mean Surface Area (µm2)",
        "Mean Volume (µm3)",
        "Mean Sphericity",
        condition,
    ]

    sns.set(style="white", context=context)
    sns.pairplot(summary[columns_of_interest], hue=condition, diag_kind="kde")
    plt.tight_layout()

    if save:
        conditions = [cond.replace("= ", "") for cond in summary[condition].unique()]
        plt.savefig("Shapes_" + "-".join(conditions), dpi=300)
    else:
        plt.show()


def all_out(tracks, condition="Condition", return_summary=False):
    """Save all plots and the tracks & summary DataFrame. Return summary."""
    plot(tracks, save=True)
    if "Sample" in tracks.columns:
        plot(tracks, plot_each_sample=True, save=True)
    plot_dr(tracks, save=True)
    joint_plot(tracks, save=True)
    lag_plot(tracks, save=True, null_model=False)
    for i, (_, cond_tracks) in enumerate(tracks.groupby(condition)):
        lag_plot(cond_tracks, save=True, skip_color=i)

    summary = summarize(tracks)
    plot_summary(summary, save=True)
    plot_uturns(summary, save=True)
    plot_shapes(summary, save=True)

    conditions = [cond.replace("= ", "") for cond in summary[condition].unique()]
    tracks.to_csv("Tracks_" + "-".join(conditions) + ".csv")
    summary.to_csv("Summary_" + "-".join(conditions) + ".csv")

    if return_summary:
        return summary


if __name__ == "__main__":
    """Demostrate motility analysis of simulated data."""
    from lana import remix

    # Uniquize & split single track
    to_uniquize = pd.DataFrame(
        {"Track_ID": 0, "Time": (0, 1, 1, 0, 2), "X": 0, "Y": 0, "Z": 0}
    )
    to_uniquize = to_uniquize.append(
        pd.DataFrame(
            {
                "Track_ID": 1,
                "Time": (0, 1, 1, 0, 2),
                "X": (0, 1, 0, 1, 0),
                "Y": 0,
                "Z": 0,
            }
        )
    )
    track_2 = pd.DataFrame(
        {"Track_ID": 2, "Time": (0, 1, 1, 1, 2), "X": 0, "Y": 0, "Z": 0}
    )
    to_uniquize = to_uniquize.append(track_2)
    _uniquize_tracks(to_uniquize, verbose=True)
    print(to_uniquize, "\n\n", track_2, "\n")

    to_split = pd.DataFrame(
        {"Track_ID": 0, "Time": np.arange(10) / 3, "X": 0, "Y": 0, "Z": 0}
    ).drop(4)
    to_split.loc[to_split.index[-2:], "X"] = 666
    _split_at_skip(to_split, 1, verbose=True)
    print(to_split)

    # Find steepest turn in single track
    track = pd.DataFrame(
        {
            "Velocity": np.ones(7) + np.sort(np.random.rand(7) / 100),
            "Turning Angle": np.sort(np.random.rand(7)) / 100,
            "Plane Angle": np.random.rand(7) / 100,
        }
    )
    track.loc[2, "Turning Angle"] = np.pi / 2
    track.loc[3, "Turning Angle"] = np.pi / 2

    tracks = remix.silly_steps(track)
    tracks["Track_ID"] = 0
    tracks["Time"] = np.arange(8)
    summary = summarize(tracks, skip_steps=2)
    plot_tracks(tracks, summary)

    # Analyze several tracks
    raw_tracks = remix.silly_tracks()
    raw_tracks.loc[:, "Time"] = raw_tracks["Time"] / 3
    plot_dr(raw_tracks)

    tracks = analyze(raw_tracks)
    tracks = tracks.drop("Z", axis=1)
    plot(tracks)
    joint_plot(tracks, skip_color=1)
    plot_tracks_parameter_space(tracks)
    plot_arrest(tracks)
    lag_plot(tracks, skip_color=1)

    summary = summarize(tracks)
    plot_summary(summary)
    plot_uturns(summary)
    plot_tracks(tracks, summary)
