"""Remix measured cell tracks"""
import numpy as np
import pandas as pd

from lana.utils import track_identifiers


def silly_steps(track_data, time_step=60, dim=3):
    """Generate a walk from velocity, turning and plane angle data"""
    steps = track_data["Velocity"].dropna().values / 60 * time_step
    turning_angles = track_data["Turning Angle"].dropna().values
    plane_angles = track_data["Plane Angle"].dropna().values
    if "Condtion" in track_data.columns:
        condition = track_data["Condition"].iloc[0] + " Rebuilt"
    else:
        condition = "Silly"
    n_steps = len(steps)

    # Walk in x-y plane w/ given velocity and turning angles
    dr = np.zeros((n_steps + 1, 3))
    dr[1, 0] = steps[0]
    for i in range(2, n_steps + 1):
        cosa = np.cos(turning_angles[i - 2])
        sina = np.sin(turning_angles[i - 2])
        dr[i, 0] = (
            (cosa * dr[i - 1, 0] - sina * dr[i - 1, 1]) * steps[i - 1] / steps[i - 2]
        )
        dr[i, 1] = (
            (sina * dr[i - 1, 0] + cosa * dr[i - 1, 1]) * steps[i - 1] / steps[i - 2]
        )

    # Add up and move 1st turn to origin
    r = np.cumsum(dr, axis=0) - dr[1, :]

    # Rotate moved positions minus the plane angles around the next step
    for i in range(2, n_steps + 1):
        r = r - dr[i, :]
        if i == n_steps:
            t = (np.random.rand() - 0.5) * 2 * np.pi
            cost = np.cos(t)
            sint = np.sin(t)
            theta = np.random.rand() * 2 * np.pi
            if dim == 3:
                phi = np.arccos(2 * np.random.rand() - 1)
            else:
                phi = np.pi
            n_vec[0] = np.sin(theta) * np.sin(phi)
            n_vec[1] = np.cos(theta) * np.sin(phi)
            n_vec[2] = np.cos(phi)
        else:
            cost = np.cos(-plane_angles[i - 2])
            sint = np.sin(-plane_angles[i - 2])
            n_vec = dr[i, :] / np.sqrt(np.sum(dr[i, :] * dr[i, :]))
        for j in range(i):
            cross_prod = np.cross(n_vec, r[j, :])
            dot_prod = np.sum(n_vec * r[j, :])
            r[j, :] = r[j, :] * cost + cross_prod * sint + n_vec * dot_prod * (1 - cost)

    r = r[0, :] - r

    track = pd.DataFrame(
        {
            "Time": np.arange(n_steps + 1) / 60 * time_step,
            "X": r[:, 0],
            "Y": r[:, 1],
            "Z": r[:, 2],
            "Source": "Silly 3D walk",
            "Condition": condition,
        }
    )

    if dim == 3:
        return track
    else:
        return track.drop("Z", axis=1)


def silly_tracks(n_tracks=100, n_steps=60):
    """Generate a DataFrame with random tracks"""
    print("\nGenerating {} random tracks".format(n_tracks))

    tracks = pd.DataFrame()
    for track_id in range(n_tracks):
        # track_data = pd.DataFrame({
        #     "Velocity": np.cumsum(np.ones(n_steps)),
        #     "Turning Angle": np.zeros(n_steps),
        #     "Plane Angle": np.zeros(n_steps),
        #     "Condition": "Trivial"})
        track_data = pd.DataFrame(
            {
                "Velocity": np.random.lognormal(0, 0.5, n_steps) * 3,
                "Turning Angle": np.random.lognormal(0, 0.5, n_steps),
                "Plane Angle": (np.random.rand(n_steps) - 0.5) * 2 * np.pi,
                "Condition": "Random",
            }
        )
        track = silly_steps(track_data)
        track["Track_ID"] = track_id
        tracks = tracks.append(track)

    return tracks


def remix_dr(tracks, n_tracks=50, n_steps=60):
    """Remix dx, dy & dz to generate new tracks (Gerard et al. 2014)"""
    time_step = int(
        np.round(next(dt for dt in sorted(tracks["Time"].diff()) if dt > 0) * 60)
    )
    print(
        "Generating {} steps from {} steps {}s apart Gerard-style.".format(
            n_tracks * n_steps, len(tracks), time_step
        )
    )

    criteria = [
        crit for crit in ["Condition", "Track_ID", "Sample"] if crit in tracks.columns
    ]

    dx = pd.DataFrame()
    for _, track in tracks.groupby(criteria):
        dx = dx.append(track[["X", "Y", "Z"]].diff())
    dx = dx.dropna()

    new_tracks = pd.DataFrame()
    for track_id in range(n_tracks):
        new_track = dx.sample(n_steps + 1, replace=True).cumsum()
        new_track["Time"] = np.arange(n_steps + 1) / 60 * time_step
        new_track["Track_ID"] = track_id
        new_tracks = new_tracks.append(new_track)

    if "Condition" in tracks.columns:
        new_tracks["Condition"] = (
            tracks["Condition"].iloc[0] + " Remixed by Gerard et al."
        )
    else:
        new_tracks["Condition"] = "Remixed by Gerard et al."

    return new_tracks.dropna().reset_index()


def remix(tracks, n_tracks=50, n_steps=60):
    """Return new tracks generated by remixing given tracks"""
    time_step = int(
        np.round(next(dt for dt in sorted(tracks["Time"].diff()) if dt > 0) * 60)
    )
    print(
        "Generating {} steps from {} steps {}s apart.".format(
            n_tracks * n_steps, len(tracks), time_step
        )
    )

    velocities_only = tracks[tracks["Turning Angle"].isnull()]["Velocity"].dropna()
    velo_and_turn = tracks[tracks["Plane Angle"].isnull()][
        ["Velocity", "Turning Angle"]
    ].dropna()
    remaining_data = tracks[["Velocity", "Turning Angle", "Plane Angle"]].dropna()

    new_tracks = pd.DataFrame()
    for i in range(n_tracks):
        track_data = velo_and_turn.sample()
        track_data = track_data.append(
            remaining_data.sample(n_steps - 2, replace=True)
        )
        track_data = track_data.append(
            pd.DataFrame(
                {
                    "Velocity": velocities_only.sample()
                }
            )
        )
        new_track = silly_steps(track_data, time_step)
        new_track["Track_ID"] = i
        new_tracks = new_tracks.append(new_track)

    if "Condition" in tracks.columns:
        new_tracks["Condition"] = tracks["Condition"].iloc[0] + " Remixed"
    else:
        new_tracks["Condition"] = "Remixed"

    return new_tracks.reset_index()


def remix_preserving_lags(tracks, n_tracks=50, n_steps=60):
    """Return new tracks preserving mean sqrd. velocity & turning angle lags"""

    def mean_lags(tracks):
        """Calculate mean lag in velocity and turning angle of track(s)"""
        means = []
        for _, track in tracks.groupby(track_identifiers(tracks)):
            lags = np.mean(track[["Velocity", "Turning Angle"]].diff() ** 2)
            if not np.isnan(lags).any():
                means.append(lags)
        return np.mean(means, axis=0)

    # Generate initial remix
    remix = tracks[["Velocity", "Turning Angle", "Plane Angle"]].dropna()
    time_step = int(
        np.round(next(dt for dt in sorted(tracks["Time"].diff()) if dt > 0) * 60)
    )
    print(
        "Generating {} steps from {} steps {}s apart, preserving lag.".format(
            n_tracks * n_steps, len(remix), time_step
        )
    )
    remix = remix.sample(n_tracks * n_steps, replace=True)
    remix["Track_ID"] = 0

    # Shuffle until mean lag is preserved
    ctrl_lags = mean_lags(tracks) * (n_tracks * n_steps - 1)
    remix_lags = mean_lags(remix) * (n_tracks * n_steps - 1)

    remix = remix.reset_index(drop=True)

    delta_lags = np.zeros(2)
    diff_lags = remix_lags - ctrl_lags
    while (diff_lags[0] > 0) or (diff_lags[1] > 0):
        index = remix.index.values
        cand = np.random.choice(index[1:-1], 2, replace=False)
        # fmt: off
        delta_lags[0] = \
            - (remix.loc[cand[0], 'Velocity'] - remix.loc[cand[0]-1, 'Velocity'])**2*(cand[0] != cand[1]+1) \
            - (remix.loc[cand[0], 'Velocity'] - remix.loc[cand[0]+1, 'Velocity'])**2*(cand[0] != cand[1]-1) \
            - (remix.loc[cand[1], 'Velocity'] - remix.loc[cand[1]-1, 'Velocity'])**2*(cand[0] != cand[1]-1) \
            - (remix.loc[cand[1], 'Velocity'] - remix.loc[cand[1]+1, 'Velocity'])**2*(cand[0] != cand[1]+1) \
            + (remix.loc[cand[1], 'Velocity'] - remix.loc[cand[0]-1, 'Velocity'])**2 \
            + (remix.loc[cand[1], 'Velocity'] - remix.loc[cand[0]+1, 'Velocity'])**2 \
            + (remix.loc[cand[0], 'Velocity'] - remix.loc[cand[1]-1, 'Velocity'])**2 \
            + (remix.loc[cand[0], 'Velocity'] - remix.loc[cand[1]+1, 'Velocity'])**2
        delta_lags[1] = \
            - (remix.loc[cand[0], 'Turning Angle'] - remix.loc[cand[0]-1, 'Turning Angle'])**2*(cand[0] != cand[1]+1) \
            - (remix.loc[cand[0], 'Turning Angle'] - remix.loc[cand[0]+1, 'Turning Angle'])**2*(cand[0] != cand[1]-1) \
            - (remix.loc[cand[1], 'Turning Angle'] - remix.loc[cand[1]-1, 'Turning Angle'])**2*(cand[0] != cand[1]-1) \
            - (remix.loc[cand[1], 'Turning Angle'] - remix.loc[cand[1]+1, 'Turning Angle'])**2*(cand[0] != cand[1]+1) \
            + (remix.loc[cand[1], 'Turning Angle'] - remix.loc[cand[0]-1, 'Turning Angle'])**2 \
            + (remix.loc[cand[1], 'Turning Angle'] - remix.loc[cand[0]+1, 'Turning Angle'])**2 \
            + (remix.loc[cand[0], 'Turning Angle'] - remix.loc[cand[1]-1, 'Turning Angle'])**2 \
            + (remix.loc[cand[0], 'Turning Angle'] - remix.loc[cand[1]+1, 'Turning Angle'])**2
        # fmt: on
        if (np.sign(delta_lags[0]) != np.sign(diff_lags[0])) and (
            np.sign(delta_lags[1]) != np.sign(diff_lags[1])
        ):
            remix_lags += delta_lags
            diff_lags += delta_lags
            index[cand[0]], index[cand[1]] = index[cand[1]], index[cand[0]]
            remix = remix.iloc[index].reset_index(drop=True)

    # Generate new tracks
    new_tracks = pd.DataFrame()
    for i in range(n_tracks):
        track_data = remix.iloc[n_steps * i : n_steps * (i + 1)].copy()
        track_data.loc[n_steps * i, "Plane Angle"] = np.nan
        new_track = silly_steps(track_data, time_step, 2 + ("Z" in tracks.columns))
        new_track["Track_ID"] = i
        new_tracks = new_tracks.append(new_track)

    if "Condition" in tracks.columns:
        new_tracks["Condition"] = (
            tracks["Condition"].iloc[0] + " Remixed preserving lags"
        )
    else:
        new_tracks["Condition"] = "Remixed preserving lags"

    assert (mean_lags(remix) <= mean_lags(tracks)).all(), "Remix did not preserve lag!"

    return new_tracks.reset_index()


if __name__ == "__main__":
    """Test & illustrate rebuilding and remixing tracks"""
    import seaborn as sns
    from lana import motility

    raw_tracks = pd.read_csv("Examples/ctrl.csv")
    ctrl = raw_tracks[raw_tracks.Track_ID == 1015.0].copy()

    # Rebuild a single track
    ctrl[["X", "Y", "Z"]] = ctrl[["X", "Y", "Z"]] - ctrl[["X", "Y", "Z"]].iloc[0]
    rebuilt = silly_steps(ctrl, time_step=20)
    rebuilt["Track_ID"] = 0
    motility.plot_tracks(ctrl.append(rebuilt))
    rebuilt = motility.analyze(rebuilt)
    print(ctrl[["Time", "Velocity", "Turning Angle", "Plane Angle"]])
    print(rebuilt[["Time", "Velocity", "Turning Angle", "Plane Angle"]])

    # Remix Ctrl
    remixed_tracks = remix(ctrl, n_tracks=1, n_steps=5)
    remixed_tracks = motility.analyze(remixed_tracks)
    print(remixed_tracks[["Time", "Velocity", "Turning Angle", "Plane Angle"]])
    print(ctrl[["Time", "Velocity", "Turning Angle", "Plane Angle"]])

    # Compare Algorithms
    remix_dr_tracks = remix_dr(raw_tracks)
    remix_tracks = remix(raw_tracks)
    remix_lags_tracks = remix_preserving_lags(raw_tracks)
    all_tracks = pd.concat(
        [remix_dr_tracks, remix_tracks, remix_lags_tracks]
    )
    tracks = motility.analyze(all_tracks)
    palette = [sns.color_palette()[i] for i in [1, 0, 2, 3]]
    motility.plot(tracks, palette=palette)
    motility.lag_plot(tracks, null_model=False)

    # Remix from short vs from long tracks
    summary = motility.summarize(raw_tracks)

    short_track_ids = [
        summary.loc[index]["Track_ID"]
        for index in summary.sort_values("Track Duration").index
        if summary["Track Duration"].sort_values().cumsum().loc[index]
        < summary["Track Duration"].sum() / 2
    ]

    short_remix = remix_preserving_lags(
        raw_tracks[raw_tracks["Track_ID"].isin(short_track_ids)], n_tracks=25, n_steps=60
    )
    long_remix = remix_preserving_lags(
        raw_tracks[~raw_tracks["Track_ID"].isin(short_track_ids)], n_tracks=25, n_steps=60
    )

    short_remix["Condition"] = "Short Tracks Remixed"
    long_remix["Condition"] = "Long Tracks Remixed"

    tracks = pd.concat([raw_tracks, short_remix, long_remix])
    # Might split some tracks as Sample-column is ignored because of NAs in remix
    tracks = motility.analyze(tracks)
    motility.plot(tracks)

    # Remix long or short tracks
    short_remix = remix_preserving_lags(raw_tracks, n_tracks=50, n_steps=300)
    long_remix = remix_preserving_lags(raw_tracks, n_tracks=25, n_steps=600)

    short_remix["Condition"] = "Short Remix"
    long_remix["Condition"] = "Long Remix"

    tracks = pd.concat([raw_tracks, short_remix, long_remix])
    tracks = motility.analyze(tracks)
    motility.plot(tracks, max_time=60)
