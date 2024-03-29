"""Analyze and plot contacts within lymph nodes"""
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.spatial as spatial
import mpl_toolkits.mplot3d.axes3d as p3
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, PathPatch
from matplotlib.ticker import MaxNLocator

from lana.utils import equalize_axis3d
from lana.utils import track_identifiers


def _find_by_distance(tracks, dcs, contact_radius, tcz_radius):
    """Find contacts among T-cell tracks and DC positions"""
    if "Appearance Time" in dcs.columns:
        available_dcs = pd.DataFrame()
    else:
        dc_tree = spatial.cKDTree(dcs[["X", "Y", "Z"]])
        available_dcs = dcs
    free_t_cells = set(tracks["Track_ID"].unique())
    contacts = pd.DataFrame()
    max_index = 0
    for time, positions in tracks.sort_values("Time").groupby("Time"):
        if "Appearance Time" not in dcs.columns:
            pass
        elif len(dcs[dcs["Appearance Time"] <= time]) == 0:
            continue
        elif len(available_dcs) != len(dcs[dcs["Appearance Time"] <= time]):
            available_dcs = dcs[dcs["Appearance Time"] <= time].reset_index()
            dc_tree = spatial.cKDTree(available_dcs[["X", "Y", "Z"]])
        positions = positions[positions["Track_ID"].isin(free_t_cells)]
        positions = positions[
            np.linalg.norm(positions[["X", "Y", "Z"]], axis=1)
            < (tcz_radius + contact_radius)
        ]
        if positions.__len__() != 0:
            t_cell_tree = spatial.cKDTree(positions[["X", "Y", "Z"]])
            new_contacts = dc_tree.query_ball_tree(t_cell_tree, contact_radius)
            for dc, dc_contacts in enumerate(new_contacts):
                for t_cell in dc_contacts:
                    contacts.loc[max_index, "Time"] = time
                    contacts.loc[max_index, "Track_ID"] = positions.iloc[t_cell][
                        "Track_ID"
                    ]
                    contacts.loc[max_index, "X"] = available_dcs.loc[dc, "X"]
                    contacts.loc[max_index, "Y"] = available_dcs.loc[dc, "Y"]
                    contacts.loc[max_index, "Z"] = available_dcs.loc[dc, "Z"]
                    max_index += 1
                    try:
                        free_t_cells.remove(positions.iloc[t_cell]["Track_ID"])
                    except KeyError:
                        print("  Warning: T cell binding two DCs.")

    if len(contacts) != 0:
        n_twice_bound = contacts["Track_ID"].duplicated().sum()
        n_twice_bound_at_same_time = contacts[["Track_ID", "Time"]].duplicated().sum()
        assert (
            n_twice_bound == n_twice_bound_at_same_time
        ), "T cells were in contacts at different times."

    return contacts


def simulate_priming(
    tracks,
    t_cell_ns=(10, 20),
    dc_ns=(10, 50),
    min_distances=(0,),
    min_dist_stds=(150 * 0,),
    contact_radii=(15 / 1.5,),
    tcz_volume=0.125e9 / 100,
    n_iter=10,
):
    """Simulate ensemble of pair-wise T cell/DC contacts within radius"""
    print(f"\nSimulating pair-wise contacts {n_iter} times")
    assert (
        max(t_cell_ns) < tracks["Track_ID"].nunique()
    ), "Max. t_cell_ns is larger than # of given tracks."

    if "Condition" not in tracks.columns:
        tracks["Condition"] = "Default"
    conditions = tracks["Condition"].unique()

    pairs = pd.DataFrame()
    for n_run in range(n_iter):
        for min_dist, min_std, cr, nt, ndc, cond in itertools.product(
            min_distances, min_dist_stds, contact_radii, t_cell_ns, dc_ns, conditions
        ):
            cond_tracks = tracks[tracks["Condition"] == cond]
            t_tracks = cond_tracks[
                cond_tracks["Track_ID"].isin(
                    np.random.choice(
                        cond_tracks["Track_ID"].unique(), nt, replace=False
                    )
                )
            ].copy()
            if min_std != 0:
                # Beware, introducing noise makes the returned pairs to not fit to the tracks,
                # e.g. in plot_details there would be a mean distance around this std!
                for track_id, track in t_tracks.groupby("Track_ID"):
                    t_tracks.loc[t_tracks["Track_ID"] == track_id, ["X", "Y", "Z"]] += (
                        np.random.randn(3) * min_std
                    )

            tcz_radius = (3 * tcz_volume / (4 * np.pi)) ** (1 / 3)
            ratio = (min_dist / tcz_radius) ** 3
            r = tcz_radius * (ratio + (1 - ratio) * np.random.rand(ndc)) ** (1 / 3)
            theta = np.random.rand(ndc) * 2 * np.pi
            phi = np.arccos(2 * np.random.rand(ndc) - 1)
            dcs = pd.DataFrame(
                {
                    "X": r * np.sin(theta) * np.sin(phi),
                    "Y": r * np.cos(theta) * np.sin(phi),
                    "Z": r * np.cos(phi),
                }
            )

            run_pairs = _find_by_distance(t_tracks, dcs, cr, tcz_radius)
            run_pairs["Run"] = n_run
            run_pairs["Cell Numbers"] = f"{nt} T cells, {ndc} DCs"
            run_pairs["T Cell Condition"] = cond
            run_pairs["Contact Radius"] = cr
            run_pairs["Minimal Initial Distance"] = min_dist
            run_pairs["Std. of Initial Position"] = min_std
            description = []
            if len(t_cell_ns) > 1 or len(conditions) > 1:
                description.append(f"{nt} {cond} T cells".replace("Default ", ""))
            if len(dc_ns) > 1:
                description.append(f"{ndc} DCs")
            if len(min_distances) > 1 or len(min_dist_stds) > 1:
                description.append(f"Min. Distance {min_dist} +/- {min_std}")
            if len(contact_radii) > 1:
                description.append(f"{cr} Contact Rad.")
            run_pairs["Description"] = ", ".join(description)
            pairs = pairs.append(run_pairs)

        print(f"  Run {n_run + 1} done.")

    # Save duration and number of runs for analysis
    pairs.reset_index(drop=True, inplace=True)
    max_index = pairs.index.max()
    pairs.loc[max_index + 1, "Time"] = tracks["Time"].max()
    pairs.loc[max_index + 1, "Run"] = n_iter - 1

    return pairs


def simulate_clustering(
    cd4_tracks,
    cd8_tracks,
    cd4_ns=(10,),
    cd8_ns=(10,),
    dc_ns=(50,),
    cd8_delays=(0,),
    contact_radii=(10,),
    focusing_factors=(1, 2, 4),
    tcz_volume=0.125e9 / 100,
    n_iter=10,
):
    """Simulate stable contacts among CD4/CD8/DCs w/ CD4 focusing CD8 on DC"""
    print(f"\nSimulating triple contacts allowing CD4/DC & CD8/DC pairs {n_iter} times")
    assert (
        max(cd4_ns) < cd4_tracks["Track_ID"].unique().__len__()
    ), "Max. cd4_ns is larger than # of given CD4+ tracks."
    assert (
        max(cd8_ns) < cd8_tracks["Track_ID"].unique().__len__()
    ), "Max. cd8_ns is larger than # of given CD8+ tracks."

    cd4_pairs = pd.DataFrame()
    cd8_pairs = pd.DataFrame()
    triples = pd.DataFrame()
    max_index = 0
    for n_run in range(n_iter):
        for cr, foc_fac, n4, n8, ndc, delay in itertools.product(
            contact_radii, focusing_factors, cd4_ns, cd8_ns, dc_ns, cd8_delays
        ):
            assert foc_fac >= 1, "Focusing Factor must be >= 1"

            description = []
            if len(cd4_ns) > 1:
                description.append(f"{n4} CD4")
            if len(cd8_delays) > 1:
                description.append(f"{n8} CD8 {delay} min. later")
            elif len(cd8_ns) > 1:
                description.append(f"{n8} CD8")
            if len(dc_ns) > 1:
                description.append(f"{ndc} DCs")
            if len(contact_radii) > 1:
                description.append(f"{cr} Contact Rad.")
            if len(focusing_factors) > 1:
                description.append(f"{foc_fac}x Focusing")

            # Create DCs
            tcz_radius = (3 * tcz_volume / (4 * np.pi)) ** (1 / 3)
            r = tcz_radius * np.random.rand(ndc) ** (1 / 3)
            theta = np.random.rand(ndc) * 2 * np.pi
            phi = np.arccos(2 * np.random.rand(ndc) - 1)
            dcs = pd.DataFrame(
                {
                    "X": r * np.sin(theta) * np.sin(phi),
                    "Y": r * np.cos(theta) * np.sin(phi),
                    "Z": r * np.cos(phi),
                }
            )

            # Find CD4-DC-Pairs
            t_tracks = cd4_tracks[
                cd4_tracks["Track_ID"].isin(
                    np.random.choice(cd4_tracks["Track_ID"].unique(), n4, replace=False)
                )
            ]
            run_cd4_pairs = _find_by_distance(t_tracks, dcs, cr, tcz_radius)
            run_cd4_pairs["Run"] = n_run
            run_cd4_pairs["Cell Numbers"] = f"{n4} CD4+ T cells, {n8} CD8+ T cells, {ndc} DCs"
            run_cd4_pairs["Contact Radius"] = cr
            run_cd4_pairs["Focusing Factor"] = foc_fac
            run_cd4_pairs["CD8 Delay"] = delay
            run_cd4_pairs["Description"] = ", ".join(description)
            cd4_pairs = cd4_pairs.append(run_cd4_pairs)

            # Find CD8-DC-Pairs
            t_tracks = cd8_tracks[
                cd8_tracks["Track_ID"].isin(
                    np.random.choice(cd8_tracks["Track_ID"].unique(), n8, replace=False)
                )
            ].copy()
            t_tracks["Time"] = t_tracks["Time"] + delay
            run_cd8_pairs = _find_by_distance(t_tracks, dcs, cr, tcz_radius)
            run_cd8_pairs["Run"] = n_run
            run_cd8_pairs["Cell Numbers"] = f"{n4} CD4+ T cells, {n8} CD8+ T cells, {ndc} DCs"
            run_cd8_pairs["Contact Radius"] = cr
            run_cd8_pairs["Focusing Factor"] = foc_fac
            run_cd8_pairs["CD8 Delay"] = delay
            run_cd8_pairs["Description"] = ", ".join(description)
            cd8_pairs = cd8_pairs.append(run_cd8_pairs)

            # Find pairs among CD8s and DCs licensed by CD4s
            if foc_fac != 1:
                for idx, dc in dcs.iterrows():
                    try:
                        dc_contacts = run_cd4_pairs[
                            np.isclose(run_cd4_pairs["X"], dc["X"])
                            & np.isclose(run_cd4_pairs["Y"], dc["Y"])
                            & np.isclose(run_cd4_pairs["Z"], dc["Z"])
                        ]
                        dcs.loc[idx, "Appearance Time"] = dc_contacts["Time"].min()
                    except KeyError:
                        continue
                dcs = dcs.dropna().reset_index(drop=True)
                lic_cd8_pairs = _find_by_distance(
                    t_tracks, dcs, cr * foc_fac, tcz_radius
                )
                lic_cd8_pairs["Run"] = n_run
                lic_cd8_pairs["Cell Numbers"] = f"{n4} CD4+ T cells, {n8} CD8+ T cells, {ndc} DCs"
                lic_cd8_pairs["Contact Radius"] = cr
                lic_cd8_pairs["CD8 Delay"] = delay
                run_cd8_pairs = run_cd8_pairs.append(lic_cd8_pairs)
                try:
                    run_cd8_pairs = run_cd8_pairs.sort_values("Time").drop_duplicates(
                        "Track_ID"
                    )
                except KeyError:
                    pass

            # Check for triples
            run_triples = pd.DataFrame()  # For assertion (and evlt. performance)
            for _, pair in run_cd8_pairs.iterrows():
                try:
                    pair_triples = run_cd4_pairs[
                        np.isclose(run_cd4_pairs["X"], pair["X"])
                        & np.isclose(run_cd4_pairs["Y"], pair["Y"])
                        & np.isclose(run_cd4_pairs["Z"], pair["Z"])
                    ]
                    closest_cd4_pair = pair_triples.loc[
                        (pair_triples["Time"] - pair["Time"]).abs().idxmin(), :
                    ]
                except (KeyError, ValueError):
                    continue
                run_triples.loc[max_index, "Track_ID"] = pair["Track_ID"]
                run_triples.loc[max_index, "CD8 Track_ID"] = pair["Track_ID"]
                run_triples.loc[max_index, "CD4 Track_ID"] = closest_cd4_pair[
                    "Track_ID"
                ]
                run_triples.loc[max_index, "Time"] = pair["Time"]
                # run_triples.loc[max_index, ['X', 'Y', 'Z']] = pair[['X', 'Y', 'Z']]
                run_triples.loc[max_index, "X"] = pair["X"]
                run_triples.loc[max_index, "Y"] = pair["Y"]
                run_triples.loc[max_index, "Z"] = pair["Z"]
                run_triples.loc[max_index, "Time Between Contacts"] = (
                    pair["Time"] - closest_cd4_pair["Time"]
                )
                run_triples.loc[max_index, "Run"] = n_run
                run_triples.loc[
                    max_index, "Cell Numbers"
                ] = f"{n4} CD4+ T cells, {n8} CD8+ T cells, {ndc} DCs"
                run_triples.loc[max_index, "Contact Radius"] = cr
                run_triples.loc[max_index, "Focusing Factor"] = foc_fac
                run_triples.loc[max_index, "CD8 Delay"] = "{delay} min. between injections"
                max_index += 1
            try:
                n_triples_of_run = len(run_triples)
            except KeyError:
                n_triples_of_run = 0
            try:
                n_cd8_pairs_of_run = len(run_cd8_pairs)
            except KeyError:
                n_cd8_pairs_of_run = 0
            assert (
                n_triples_of_run <= n_cd8_pairs_of_run
            ), "More triples found than possible."
            for _, triple in run_triples.iterrows():
                cd8_position = cd8_tracks[
                    (cd8_tracks["Track_ID"] == triple["CD8 Track_ID"])
                    & (cd8_tracks["Time"] == triple["Time"])
                ][["X", "Y", "Z"]]
                cd4_contact_time = triple["Time"] - triple["Time Between Contacts"]
                cd4_position = cd4_tracks[
                    (cd4_tracks["Track_ID"] == triple["CD4 Track_ID"])
                    & np.isclose(cd4_tracks["Time"], cd4_contact_time)
                ][["X", "Y", "Z"]]
                distance = np.linalg.norm(cd4_position.values - cd8_position.values)
                assert distance <= cr * (1 + foc_fac), "Triple too far apart."
            run_triples["Description"] = ", ".join(description)
            triples = triples.append(run_triples)

        print(f"  Run {n_run + 1} done.")

    # Save duration and number of runs for analysis
    for df, tracks in zip(
        [cd4_pairs, cd8_pairs, triples], [cd4_tracks, cd8_tracks, cd4_tracks]
    ):
        df.reset_index(drop=True, inplace=True)
        max_index = df.index.max()
        df.loc[max_index + 1, "Time"] = tracks["Time"].max()
        df.loc[max_index + 1, "Run"] = n_iter - 1

    return {"CD4-DC-Pairs": cd4_pairs, "CD8-DC-Pairs": cd8_pairs, "Triples": triples}


def plot_details(contacts, tracks=None, parameters="Description", context="notebook"):
    """Plot distances over time, time in contact and time vs. distance to 0"""
    sns.set(style="ticks", context=context)
    if tracks is not None:
        _, axes = plt.subplots(ncols=3, figsize=(12, 6))
        axes[0].set_xlabel("Time [min]")
        axes[0].set_ylabel(r"Distance [$\mu$m]")
        axes[1].set_xlabel("Time within Contact Radius [min]")
        axes[1].set_ylabel("Number of Contacts")
        axes[2].set_xlabel("Contact Time [h]")
        axes[2].set_ylabel("Distance from Origin")
    else:
        plt.gca().set_xlabel("Contact Time [h]")
        plt.gca().set_ylabel("Distance from Origin")

    contacts = contacts.dropna(axis=1, how="all").copy()
    for i, (cond, cond_contacts) in enumerate(contacts.groupby(parameters)):
        color = sns.color_palette(n_colors=i + 1)[-1]
        if tracks is not None:
            if cond_contacts["Contact Radius"].dropna().nunique() != 1:
                raise ValueError("Condition with more than one contact radius")
            radius = cond_contacts["Contact Radius"].max()
            distances = pd.Series(dtype=float)
            durations = []
            for _, contact in cond_contacts.dropna().iterrows():
                track = tracks[tracks["Track_ID"] == contact["Track_ID"]]
                track = track[["Time", "X", "Y", "Z"]]
                track = track[track["Time"] <= contact["Time"] + 20]
                track = track[track["Time"] >= contact["Time"] - 10]
                distance = pd.Series(
                    np.linalg.norm(
                        track[["X", "Y", "Z"]].astype(float)
                        - contact[["X", "Y", "Z"]].astype(float),
                        axis=1,
                    ),
                    track["Time"] - contact["Time"],
                )
                time_step = track["Time"].diff().mean()
                distances = distances.append(distance)
                durations.append(distance[distance <= radius].size * time_step)

            distances.index = np.round(distances.index, 5)  # Handle non-integer 'Times'
            distats = distances.groupby(distances.index).describe()
            axes[0].plot(distats.index, distats["50%"], color=color)
            axes[0].fill_between(
                distats.index, distats["25%"], distats["75%"], color=color, alpha=0.2
            )
            axes[0].fill_between(
                distats.index, distats["min"], distats["max"], color=color, alpha=0.2
            )

            sns.histplot(
                durations,
                bins=np.arange(20 + 1),
                kde=False,
                common_norm=True,
                ax=axes[1],
                color=color,
                fill=False,
            )

        if tracks is not None:
            ax = axes[2]
        else:
            ax = plt.gca()
        ax.scatter(
            cond_contacts["Time"] / 60,
            np.linalg.norm(cond_contacts[["X", "Y", "Z"]].astype(np.float64), axis=1),
            color=color,
            label=cond,
        )
        ax.legend(loc=4)

    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_numbers(
    contacts,
    parameters="Description",
    t_detail=1,
    palette="deep",
    save=False,
    context="notebook",
):
    """Plot accumulation and final number of T cells in contact with DC"""
    t_cells_in_contact = contacts.drop_duplicates(["Track_ID", "Run", parameters])

    sns.set(style="ticks", palette=palette, context=context)

    _ = plt.figure(figsize=(8, 5.5))
    n_parameter_sets = len(t_cells_in_contact[parameters].unique()) - 1  # nan for t_end
    gs = gridspec.GridSpec(n_parameter_sets, 2)
    detail_ax = plt.subplot(gs[:, 0])
    ax0 = plt.subplot(gs[1])

    t_max = t_cells_in_contact["Time"].max()
    if t_detail > t_max:
        t_detail = t_max
    detail_ax.set_ylabel(f"Distribution of T Cells in Contact at {t_detail}h")

    final_sum = t_cells_in_contact.groupby(parameters).count()["Time"]
    order = list(final_sum.sort_values().index.values)[::-1]

    if context == "talk":
        size = "small"
    else:
        size = "medium"

    for label, _contacts in t_cells_in_contact.groupby(parameters):
        i = order.index(label)
        n_runs = t_cells_in_contact["Run"].max() + 1
        label = "  " + str(label) + " (n = {:.0f})".format(n_runs)
        detail_ax.text(i * 2 - 0.5, 0, label, rotation=90, va="bottom", fontsize=size)

        if i == 0:
            dynamic_ax = ax0
            dynamic_ax.set_yticks([0, 50, 100])
        else:
            dynamic_ax = plt.subplot(gs[2 * i + 1], sharex=ax0, sharey=ax0)
        dynamic_ax.set_rasterization_zorder(0)

        if (t_max % (4 * 60) == 0) and (t_max // (4 * 60) > 1):
            dynamic_ax.set_xticks([4 * i for i in range(int(t_max // 4) + 1)])

        if i < n_parameter_sets - 1:
            plt.setp(dynamic_ax.get_xticklabels(), visible=False)
        else:
            dynamic_ax.set_xlabel("Time [h]")

        if t_detail < t_max / 60:
            dynamic_ax.axvline(t_detail, c="0", ls=":")

        color = sns.color_palette(n_colors=i + 1)[-1]

        accumulation = (
            _contacts[["Run", "Time"]]
            .pivot_table(columns="Run", index="Time", aggfunc=len, fill_value=0)
            .cumsum()
        )
        runs_with_n_contacts = accumulation.apply(
            lambda x: x.value_counts(), axis=1
        ).fillna(0)
        runs_with_n_contacts = runs_with_n_contacts[runs_with_n_contacts.columns[::-1]]
        runs_with_geq_n_contacts = runs_with_n_contacts.cumsum(axis=1)
        runs_with_geq_n_contacts.loc[t_max, :] = runs_with_geq_n_contacts.iloc[-1]
        detail_runs = runs_with_geq_n_contacts[
            runs_with_geq_n_contacts.index <= t_detail * 60
        ]

        for n_contacts in [n for n in runs_with_geq_n_contacts.columns if n > 0]:
            dynamic_ax.fill_between(
                runs_with_geq_n_contacts[n_contacts].index / 60,
                0,
                runs_with_geq_n_contacts[n_contacts].values / n_runs * 100,
                color=color,
                alpha=1 / runs_with_n_contacts.columns.max(),
                zorder=-1,
            )

            percentage = detail_runs[n_contacts].iloc[-1] / n_runs * 100
            detail_ax.bar(
                i * 2 + 0.38,
                percentage,
                color=color,
                alpha=1 / runs_with_n_contacts.columns.max(),
                zorder=-1,
            )

            if n_contacts == detail_runs.columns.max():
                next_percentage = 0
            else:
                next_n = next(n for n in detail_runs.columns[::-1] if n > n_contacts)
                next_percentage = detail_runs[next_n].iloc[-1] / n_runs * 100

            percentage_diff = percentage - next_percentage
            if percentage_diff > 3:
                detail_ax.text(
                    i * 2 + 0.38,
                    percentage - percentage_diff / 2 - 0.5,
                    int(n_contacts),
                    ha="center",
                    va="center",
                    fontsize=size,
                )

    detail_ax.set_xlim(left=-0.8)
    detail_ax.set_xticks([])
    detail_ax.set_yticks([0, 25, 50, 75, 100])
    detail_ax.set_ylim([0, 100])
    dynamic_ax.set_ylim([0, 100])
    dynamic_ax.set_xlim(left=0)
    detail_ax.set_rasterization_zorder(0)

    sns.despine()
    plt.tight_layout()

    if save == True:
        save = "numbers.png"

    if save:
        plt.savefig(save, dpi=300)
    else:
        plt.show()


def plot_percentage(
    contacts,
    parameters="Description",
    t_detail=1,
    n_t_cells=100,
    save=False,
    palette="deep",
    context="notebook",
):
    """Plot final percentage of T cells in contact with DC"""
    t_cells_in_contact = contacts.drop_duplicates(["Track_ID", "Run", parameters])
    contacts_at_t_detail = t_cells_in_contact[
        t_cells_in_contact["Time"] <= t_detail * 60
    ]

    sns.set(style="ticks", palette=palette, context=context)

    total_contacts = contacts_at_t_detail[["Run", parameters]].pivot_table(
        columns=parameters, index="Run", aggfunc=len, fill_value=0
    )

    normalized_contacts = total_contacts / n_t_cells * 100

    sorted_contacts = normalized_contacts.reindex(
        sorted(total_contacts.columns, key=lambda col: total_contacts[col].median()),
        axis=1,
    )

    ax = sns.violinplot(data=sorted_contacts, cut=0, inner=None, bw=0.75)
    ax.set_xlabel("")
    ax.set_ylabel("% T cells in contact")
    plt.xticks(rotation=45, horizontalalignment="right")

    sns.despine()
    plt.tight_layout()
    plt.show()

    if save == True:
        save = "raw_violins.csv"

    if save:
        sorted_contacts.to_csv(save)


def plot_triples(pairs_and_triples, parameters="Description", context="notebook"):
    """Plot # of CD8+ T cells in triples and times between 1st and 2nd contact"""
    cd8_in_triples = pairs_and_triples["Triples"].drop_duplicates(
        ["CD8 Track_ID", "Run", parameters]
    )
    cd8_in_pairs = (
        pairs_and_triples["CD8-DC-Pairs"]
        .drop_duplicates(["Track_ID", "Run", parameters])
        .copy()
    )
    cd8_in_pairs["CD8 Track_ID"] = cd8_in_pairs["Track_ID"]
    cd8_activated = cd8_in_pairs.append(cd8_in_triples).drop_duplicates(
        ["CD8 Track_ID", "Run", parameters]
    )

    sns.set(style="ticks", context=context)

    _, (activ_ax, triples_ax, timing_ax) = plt.subplots(ncols=3, figsize=(12, 5.5))

    activ_ax.set_ylabel("Percentage of Final Activated CD8+ T Cells")
    triples_ax.set_ylabel("Percentage of Final CD8+ T Cells in Triples")
    timing_ax.set_ylabel("Time Between Contacts")
    timing_ax.set_yticks([])

    final_sum = cd8_activated.groupby(parameters).count()["Time"]
    order = list(final_sum.sort_values().index.values)

    for label, _triples in cd8_activated.groupby(parameters):
        i = order.index(label)
        n_runs = cd8_in_triples["Run"].max() + 1
        label = "  " + str(label) + " (n = {:.0f})".format(n_runs)
        activ_ax.text(i * 2 - 0.5, 0, label, rotation=90, va="bottom")

        color = sns.color_palette(n_colors=i + 1)[-1]

        accumulation = (
            _triples[["Run", "Time"]]
            .pivot_table(columns="Run", index="Time", aggfunc=len, fill_value=0)
            .cumsum()
        )
        runs_with_n_contacts = accumulation.apply(
            lambda x: x.value_counts(), axis=1
        ).fillna(0)
        runs_with_n_contacts = runs_with_n_contacts[runs_with_n_contacts.columns[::-1]]
        runs_with_geq_n_contacts = runs_with_n_contacts.cumsum(axis=1)
        runs_with_geq_n_contacts.loc[
            cd8_in_triples["Time"].max(), :
        ] = runs_with_geq_n_contacts.iloc[-1]

        for n_contacts in [n for n in runs_with_geq_n_contacts.columns if n > 0]:
            percentage = runs_with_geq_n_contacts[n_contacts].iloc[-1] / n_runs * 100
            activ_ax.bar(
                i * 2 + 0.38,
                percentage,
                color=color,
                alpha=1 / runs_with_n_contacts.columns.max(),
            )

            if n_contacts == runs_with_geq_n_contacts.columns.max():
                next_percentage = 0
            else:
                next_n = next(
                    n for n in runs_with_geq_n_contacts.columns[::-1] if n > n_contacts
                )
                next_percentage = (
                    runs_with_geq_n_contacts[next_n].iloc[-1] / n_runs * 100
                )

            percentage_diff = percentage - next_percentage
            if percentage_diff > 3:
                activ_ax.text(
                    i * 2 + 0.38,
                    percentage - percentage_diff / 2 - 0.5,
                    int(n_contacts),
                    ha="center",
                    va="center",
                )

    for label, _triples in cd8_in_triples.groupby(parameters):
        i = order.index(label)
        n_runs = cd8_in_triples["Run"].max() + 1
        label = "  " + str(label) + " (n = {:.0f})".format(n_runs)
        triples_ax.text(i * 2 - 0.5, 0, label, rotation=90, va="bottom")

        color = sns.color_palette(n_colors=i + 1)[-1]

        accumulation = (
            _triples[["Run", "Time"]]
            .pivot_table(columns="Run", index="Time", aggfunc=len, fill_value=0)
            .cumsum()
        )
        runs_with_n_contacts = accumulation.apply(
            lambda x: x.value_counts(), axis=1
        ).fillna(0)
        runs_with_n_contacts = runs_with_n_contacts[runs_with_n_contacts.columns[::-1]]
        runs_with_geq_n_contacts = runs_with_n_contacts.cumsum(axis=1)
        runs_with_geq_n_contacts.loc[
            cd8_in_triples["Time"].max(), :
        ] = runs_with_geq_n_contacts.iloc[-1]

        for n_contacts in [n for n in runs_with_geq_n_contacts.columns if n > 0]:
            percentage = runs_with_geq_n_contacts[n_contacts].iloc[-1] / n_runs * 100
            triples_ax.bar(
                i * 2 + 0.38,
                percentage,
                color=color,
                alpha=1 / runs_with_n_contacts.columns.max(),
            )

            if n_contacts == runs_with_geq_n_contacts.columns.max():
                next_percentage = 0
            else:
                next_n = next(
                    n for n in runs_with_geq_n_contacts.columns[::-1] if n > n_contacts
                )
                next_percentage = (
                    runs_with_geq_n_contacts[next_n].iloc[-1] / n_runs * 100
                )

            percentage_diff = percentage - next_percentage
            if percentage_diff > 3:
                triples_ax.text(
                    i * 2 + 0.38,
                    percentage - percentage_diff / 2 - 0.5,
                    int(n_contacts),
                    ha="center",
                    va="center",
                )

        bins = (
            np.arange(
                cd8_in_triples["Time Between Contacts"].min(),
                cd8_in_triples["Time Between Contacts"].max(),
                15,
            )
            / 60
        )
        sns.histplot(
            _triples["Time Between Contacts"] / 60,
            kde=False,
            bins=bins,
            common_norm=True,
            color=color,
            fill=False,
            ax=timing_ax,
        )
        timing_ax.set_xlabel("Time [h]")
        timing_ax.set_ylabel("")

    triples_ax.set_xlim(left=-0.8)
    triples_ax.set_xticks([])
    triples_ax.set_yticks([0, 25, 50, 75, 100])
    triples_ax.set_ylim([0, 100])

    activ_ax.set_xlim(left=-0.8)
    activ_ax.set_xticks([])
    activ_ax.set_yticks([0, 25, 50, 75, 100])
    activ_ax.set_ylim([0, 100])

    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_triples_vs_pairs(triples, parameters="Description", context="notebook"):
    """Scatter plot pure CD8-DC-Pairs vs Triples per run"""
    pairs = triples["CD8-DC-Pairs"]
    triples = triples["Triples"]

    contact_numbers = pd.DataFrame()
    max_index = 0
    for run, par in itertools.product(
        range(int(pairs["Run"].max()) + 1), pairs[parameters].dropna().unique()
    ):
        contact_numbers.loc[max_index, "Run"] = run
        contact_numbers.loc[max_index, "Parameter"] = par
        cd8_in_triples = set(
            triples[(triples["Run"] == run) & (triples[parameters] == par)][
                "CD8 Track_ID"
            ]
        )
        contact_numbers.loc[max_index, "# CD8 in Triples"] = len(cd8_in_triples)
        cd8_in_pairs = set(
            pairs[(pairs["Run"] == run) & (pairs[parameters] == par)]["Track_ID"]
        )
        contact_numbers.loc[max_index, "# CD8 in Pairs"] = len(
            cd8_in_pairs.difference(cd8_in_triples)
        )
        max_index += 1

    sns.set(style="ticks", context=context)
    # sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    _, axes = plt.subplots(ncols=2, figsize=(11, 5.5))
    axes[0].set_xlabel("# CD8 in Triples")
    axes[0].set_ylabel("# CD8 in Pairs")
    axes[1].set_xlabel("arctan of # Triples/# Pairs")
    axes[1].set_ylabel("Numbers of Simulations")
    legend = []
    for i, (par, numbers) in enumerate(contact_numbers.groupby("Parameter")):
        color = sns.color_palette(n_colors=i + 1)[-1]
        axes[0].scatter(
            numbers["# CD8 in Triples"] + np.random.rand(len(numbers)) / 2,
            numbers["# CD8 in Pairs"] + np.random.rand(len(numbers)) / 2,
            color=color,
        )
        ratios = np.arctan(numbers["# CD8 in Triples"] / numbers["# CD8 in Pairs"])
        sns.histplot(
            ratios,
            color=color,
            ax=axes[1],
            bins=np.arange(21) * np.pi / 40,
            fill=False,
        )
        legend.append(par)
    axes[0].legend(legend, frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_triples_ratio(
    triples, parameters="Description", order=None, context="notebook"
):
    """Plot #triples/(#triples + #doublets)/(#licensedDCs/#DCs)"""
    pairs = triples["CD8-DC-Pairs"]
    licensed = triples["CD4-DC-Pairs"]
    triples = triples["Triples"]

    ratios = pd.DataFrame()
    max_index = 0
    for run, par in itertools.product(
        range(int(pairs["Run"].max()) + 1), pairs[parameters].dropna().unique()
    ):
        _pairs = pairs[(pairs["Run"] == run) & (pairs[parameters] == par)]
        _licensed = licensed[(licensed["Run"] == run) & (licensed[parameters] == par)]
        _triples = triples[(triples["Run"] == run) & (triples[parameters] == par)]
        # More triples than pairs possible if foc_fac > 1! Thus sets ...
        cd8_in_triples = set(_triples["CD8 Track_ID"])
        n_cd8_in_pairs_or_triples = len(cd8_in_triples.union(set(_pairs["Track_ID"])))
        n_cd8_in_triples = len(cd8_in_triples)
        n_lic_dcs = len(_licensed["X"].unique())
        if n_cd8_in_pairs_or_triples > 0 and n_lic_dcs > 0:
            try:
                cell_numbers = _triples["Cell Numbers"].iloc[0]
            except IndexError:
                cell_numbers = _pairs["Cell Numbers"].iloc[0]
            n_dcs = int(
                next(sub for sub in cell_numbers.split()[::-1] if sub.isdigit())
            )
            ratios.loc[max_index, "Triple Ratio"] = (
                n_cd8_in_triples / n_cd8_in_pairs_or_triples
            ) / (n_lic_dcs / n_dcs)
            ratios.loc[max_index, "Run"] = run
            ratios.loc[max_index, parameters] = par
            max_index += 1
        if n_cd8_in_pairs_or_triples > 0:
            try:
                cell_numbers = _triples["Cell Numbers"].iloc[0]
            except IndexError:
                cell_numbers = _pairs["Cell Numbers"].iloc[0]
            n_CD8 = int(cell_numbers.split()[4])
            ratios.loc[max_index, "CD8 Ratio"] = n_cd8_in_pairs_or_triples / n_CD8
        else:
            ratios.loc[max_index, "CD8 Ratio"] = 0
        ratios.loc[max_index, "Run"] = run
        ratios.loc[max_index, parameters] = par
        max_index += 1

    sns.set(style="ticks", context=context)
    _, axes = plt.subplots(1, 2, figsize=(8, 5.5))
    sns.boxplot(
        x="Triple Ratio",
        y=parameters,
        data=ratios,
        notch=False,
        order=order,
        ax=axes[0],
    )
    sns.stripplot(
        x="Triple Ratio",
        y=parameters,
        data=ratios,
        jitter=True,
        color="0.3",
        size=1,
        order=order,
        ax=axes[0],
    )
    sns.boxplot(
        x="CD8 Ratio", y=parameters, data=ratios, notch=False, order=order, ax=axes[1]
    )
    sns.stripplot(
        x="CD8 Ratio",
        y=parameters,
        data=ratios,
        jitter=True,
        color="0.3",
        size=1,
        order=order,
        ax=axes[1],
    )
    axes[0].axvline(1, c="0", ls=":")
    axes[0].set_xlabel(
        r"$\frac{\mathrm{Triples}/\mathrm{Activated}}"
        "{\mathrm{Licensed}/\mathrm{Total}}$",
        fontsize=15,
    )
    axes[0].set_ylabel("")
    axes[1].set_xlabel("Activated CD8/Total CD8")
    axes[1].set_ylabel("")
    axes[1].get_yaxis().set_visible(False)
    sns.despine(trim=True)
    sns.despine(ax=axes[1], top=True, right=True, left=True, bottom=False, trim=True)
    plt.tight_layout()
    plt.show()


def plot_situation(
    tracks,
    n_tracks=6 * 3,
    n_dcs=50,
    tcz_volume=0.524e9 / 400,
    min_distance=0,
    min_distance_std=200 / 10,
    zoom=1,
    t_detail=None,
    save=False,
    context="notebook",
):
    """Plot some T cell tracks, DC positions and T cell zone volume"""
    sns.set(style="ticks", context=context)

    _ = plt.figure(figsize=(8, 5.5))
    gs = gridspec.GridSpec(2, 3)
    space_ax = plt.subplot(gs[:, :-1], projection="3d")
    time_ax = plt.subplot(gs[0, -1])
    reach_ax = plt.subplot(gs[1, -1])
    plt.locator_params(nbins=6)

    space_ax.set_title(f"{n_tracks} T Cell Tracks & {n_dcs} DCs")

    n_conditions = len(tracks["Condition"].unique())
    palette = itertools.cycle(sns.color_palette())

    if min_distance_std != 0:
        moved_tracks = tracks.copy()
        for id in tracks["Track_ID"].unique():
            moved_tracks.loc[moved_tracks["Track_ID"] == id, ["X", "Y", "Z"]] += (
                np.random.randn(3) * min_distance_std
            )
    else:
        moved_tracks = tracks

    for i, (cond, cond_tracks) in enumerate(moved_tracks.groupby("Condition")):
        choice = np.random.choice(
            cond_tracks["Track_ID"].unique(), int(n_tracks / n_conditions)
        )
        chosen_tracks = cond_tracks[cond_tracks["Track_ID"].isin(choice)]
        for _, track in chosen_tracks.groupby(track_identifiers(chosen_tracks)):
            if t_detail:
                track = track[track["Time"] <= t_detail * 60]
            if n_conditions > 1:
                color = sns.color_palette(n_colors=i + 1)[-1]
            else:
                color = next(palette)
            space_ax.plot(
                track["X"].values, track["Y"].values, track["Z"].values, color=color
            )

    tcz_radius = (3 * tcz_volume / (4 * np.pi)) ** (1 / 3)
    ratio = (min_distance / tcz_radius) ** 3
    r = tcz_radius * (ratio + (1 - ratio) * np.random.rand(n_dcs)) ** (1 / 3)
    theta = np.random.rand(n_dcs) * 2 * np.pi
    phi = np.arccos(2 * np.random.rand(n_dcs) - 1)
    dcs = pd.DataFrame(
        {
            "X": r * np.sin(theta) * np.sin(phi),
            "Y": r * np.cos(theta) * np.sin(phi),
            "Z": r * np.cos(phi),
        }
    )
    space_ax.scatter(dcs["X"], dcs["Y"], dcs["Z"], c="y")

    r = (3 * tcz_volume / (4 * np.pi)) ** (1 / 3)
    for i in ["x", "y", "z"]:
        circle = Circle((0, 0), r, fill=False, linewidth=2)
        space_ax.add_patch(circle)
        art3d.pathpatch_2d_to_3d(circle, z=0, zdir=i)

    time_ax.set_xlabel("Time within Lymph Node [h]")
    time_ax.set_ylabel("Probab. Density")

    reach_ax.set_xlabel(r"Maximal Reach [$\mu$m]")
    reach_ax.set_ylabel("Probab. Density")

    def residence_time(track):
        return (
            track["Time"].diff().mean()
            / 60
            * len(track[np.linalg.norm(track[["X", "Y", "Z"]], axis=1) < r])
        )

    for i, (cond, cond_tracks) in enumerate(moved_tracks.groupby("Condition")):
        color = sns.color_palette(n_colors=i + 1)[-1]
        residence_times = [
            residence_time(track) for _, track in cond_tracks.groupby("Track_ID")
        ]
        if not all(time == residence_times[0] for time in residence_times):
            sns.histplot(
                residence_times,
                kde=False,
                common_norm=True,
                ax=time_ax,
                label=cond,
                color=color,
            )
        max_reaches = [
            max(np.linalg.norm(track[["X", "Y", "Z"]], axis=1))
            for _, track in cond_tracks.groupby("Track_ID")
        ]
        sns.histplot(
            max_reaches, kde=False, common_norm=True, ax=reach_ax, label=cond, color=color
        )

    time_ax.set_yticks([])
    time_ax.axvline(np.median(residence_times), c="0", ls=":")
    sns.despine(ax=time_ax)
    reach_ax.set_yticks([])
    reach_ax.legend(frameon=False)
    reach_ax.axvline(tcz_radius, c="0", ls=":")
    sns.despine(ax=reach_ax)
    equalize_axis3d(space_ax, zoom)
    plt.tight_layout()

    if save == True:
        save = "situation.png"

    if save:
        plt.savefig(save, dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    from lana.remix import silly_tracks

    tracks = silly_tracks(25, 180)
    tracks["Time"] = tracks["Time"] / 3
    plot_situation(tracks, n_tracks=10, n_dcs=200, min_distance=60)

    pairs = simulate_priming(tracks)
    plot_details(pairs, tracks)
    plot_numbers(pairs)
    plot_percentage(pairs, n_t_cells=[10, 10, 20, 20])

    pairs_and_triples = simulate_clustering(tracks, tracks)
    plot_details(pairs_and_triples["CD8-DC-Pairs"], tracks)
    plot_details(pairs_and_triples["Triples"])
    plot_numbers(pairs_and_triples["CD8-DC-Pairs"])
    plot_numbers(pairs_and_triples["Triples"])
    plot_triples(pairs_and_triples)
    plot_triples_vs_pairs(pairs_and_triples)
    plot_triples_ratio(pairs_and_triples)
