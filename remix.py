"""Remix measured cell tracks"""
import numpy as np
import pandas as pd


def silly_3d_steps(track_data=None, n_steps=25):
    """Generate a walk from track data (i.e. velocities, turning & rolling angles)"""
    if type(track_data) != pd.core.frame.DataFrame:
        # velocities = np.cumsum(np.ones(n_steps))
        # turning_angles = np.zeros(n_steps-1)
        # rolling_angles = np.zeros(n_steps-2)
        velocities = np.random.lognormal(0, 0.5, n_steps)
        turning_angles = np.random.lognormal(0, 0.5, n_steps-1)
        rolling_angles = (np.random.rand(n_steps-2) - 0.5)*2*np.pi
        condition = 'Random'
    else:
        velocities = track_data['Velocity'].dropna().values
        turning_angles = track_data['Turning Angle'].dropna().values
        rolling_angles = track_data['Rolling Angle'].dropna().values
        if 'Condtion' in track_data.columns:
            condition = track_data['Condition'].iloc[0] + ' Rebuilt'
        else:
            condition = 'Rebuilt'
        n_steps = velocities.__len__()

    # Walk in x-y plane w/ given velocity and turning angles
    dr = np.zeros((n_steps+1, 3))
    dr[1,0] = velocities[0]
    for i in range(2, n_steps+1):
        cosa = np.cos(turning_angles[i-2])
        sina = np.sin(turning_angles[i-2])
        dr[i,0] = (cosa*dr[i-1,0] - sina*dr[i-1,1])*velocities[i-1]/velocities[i-2]
        dr[i,1] = (sina*dr[i-1,0] + cosa*dr[i-1,1])*velocities[i-1]/velocities[i-2]

    # Add up and move 1st turn to origin
    r = np.cumsum(dr, axis=0) - dr[1,:]

    # Rotate moved positions minus the rolling angles around the next step
    for i in range(2, n_steps+1):
        r = r - dr[i,:]
        if i == n_steps:
            t = (np.random.rand() - 0.5)*2*np.pi
            cost = np.cos(t)
            sint = np.sin(t)
            theta = np.random.rand()*2*np.pi
            phi = np.arccos(2*np.random.rand() - 1)
            n_vec[0] = np.sin(theta)*np.sin(phi)
            n_vec[1] = np.cos(theta)*np.sin(phi)
            n_vec[2] = np.cos(phi)
        else:
            cost = np.cos(-rolling_angles[i-2])
            sint = np.sin(-rolling_angles[i-2])
            n_vec = dr[i,:]/np.sqrt(np.sum(dr[i,:]*dr[i,:]))
        for j in range(i):
            cross_prod = np.cross(n_vec, r[j,:])
            dot_prod = np.sum(n_vec*r[j,:])
            r[j,:] = r[j,:]*cost + cross_prod*sint + n_vec*dot_prod*(1 - cost)

    return pd.DataFrame({'Time': np.arange(n_steps+1), 'X': -r[:,0], 'Y': -r[:,1],
        'Z': -r[:,2], 'Source': 'Silly 3D walk', 'Condition': condition})


def silly_tracks(ntracks=25):
    """Generates a DataFrame with random tracks"""
    tracks = pd.DataFrame()
    for track_id in range(ntracks):
        track = silly_3d_steps()
        track['Track_ID'] = track_id
        tracks = tracks.append(track)

    return tracks


def remidx(tracks, n_tracks=50, n_steps=60):
    """Remix dx, dy & dz to generate new tracks (Gerard et al. 2014)"""
    criteria = [crit for crit in ['Condition', 'Track_ID', 'Sample']
        if crit in tracks.columns]

    dx = pd.DataFrame()
    for _, track in tracks.groupby(criteria):
        dx = dx.append(track[['X', 'Y', 'Z']].diff())
    dx = dx.dropna()

    new_tracks = pd.DataFrame()
    for track_id in range(n_tracks):
        new_track = dx.ix[np.random.choice(dx.index, n_steps+1)].cumsum()
        new_track['Time'] = np.arange(n_steps+1)
        new_track['Track_ID'] = track_id
        new_tracks = new_tracks.append(new_track)

    if 'Condition' in tracks.columns:
        new_tracks['Condition'] = tracks['Condition'].iloc[0] + ' Remixed by Gerard et al.'
    else:
        new_tracks['Condition'] = 'Remixed by Gerard et al.'

    return new_tracks.dropna().reset_index()


def sample_dr(tracks, n_tracks=50, n_steps=60):
    """Sample from dr_i based on last"""
    import statsmodels.api as sm
    # Learn KDE
    dep_data = tracks[tracks['Track Time'] != 0]
    criteria = [crit for crit in ['Condition', 'Sample', 'Track_ID']
        if crit in tracks.columns]
    indep_data = pd.DataFrame()
    for _, track in tracks.groupby(criteria):
        indep_data = indep_data.append(track.iloc[:-1])

    next_step_kde = sm.nonparametric.KDEMultivariateConditional(
        dep_data[['X', 'Y', 'Z']].diff().stack().values,
        indep_data[['X', 'Y', 'Z']].diff().stack().values,
        dep_type='c', indep_type='c', bw='normal_reference')
    max_kde = max(next_step_kde.pdf())

    # Generate new tracks
    new_tracks = pd.DataFrame()
    for track_id in range(n_tracks):
        track = tracks.ix[np.random.choice(tracks.index.values, 1)] \
            [['X', 'Y', 'Z']]
        max_index = max(track.index)
        while track.__len__() < n_steps+1:
            candidate = [
                np.random.rand()*np.percentile(initial_data['Velocity'], 99.5),
                np.random.rand()*np.pi]
            r = np.random.rand()
            if next_step_kde.pdf(candidate,
                track.loc[max_index, ['Velocity', 'Turning Angle']]) > max_kde*r:
                max_index = max_index+1
                track = track.append(pd.DataFrame({'Velocity': candidate[0],
                    'Turning Angle': candidate[1]}, index=[max_index]))

    #     new_track = new_track.cumsum()
    #     new_track['Track_ID'] = track_id
    #     new_tracks = new_tracks.append(new_track)
    #
    # if 'Condition' in tracks.columns:
    #     new_tracks['Condition'] = tracks['Condition'].iloc[0] + ' Sampled'
    # else:
    #     new_tracks['Condition'] = 'Sampled'
    #
    # return new_tracks.reset_index()


def remix(tracks=None, n_tracks=50, n_steps=60):
    """Return new tracks generated by remixing given tracks"""
    print('Generating  {} steps from {} steps.'.format(
        n_tracks*n_steps, tracks.__len__()))

    velocities_only = tracks[tracks['Turning Angle'].isnull()] \
        ['Velocity'].dropna()
    velo_and_turn = tracks[tracks['Rolling Angle'].isnull()] \
        [['Velocity', 'Turning Angle']].dropna()
    remaining_data = tracks[['Velocity', 'Turning Angle', 'Rolling Angle']].dropna()

    new_tracks = pd.DataFrame()
    for i in range(n_tracks):
        track_data = velo_and_turn.ix[np.random.choice(velo_and_turn.index.values, 1)]
        track_data = track_data.append(
            remaining_data.ix[np.random.choice(remaining_data.index.values, n_steps-2)])
        track_data = track_data.append(pd.DataFrame({'Velocity':
            velocities_only[np.random.choice(velocities_only.index.values, 1)]}))
        new_track = silly_3d_steps(track_data)
        new_track['Track_ID'] = i
        new_tracks = new_tracks.append(new_track)

    if 'Condition' in tracks.columns:
        new_tracks['Condition'] = tracks['Condition'].iloc[0] + ' Remixed'
    else:
        new_tracks['Condition'] = 'Remixed'

    return new_tracks.reset_index()


def remix_preserving_lags(tracks, n_tracks=50, n_steps=60):
    """Return new tracks preserving mean sqrd. velocity & turning angle lags"""
    def mean_lags(tracks):
        """Calculate mean lag in velocity of track(s)"""
        criteria = [crit for crit in ['Condition', 'Sample', 'Track_ID']
            if crit in tracks.columns]
        means = []
        for _, track in tracks.groupby(criteria):
            means.append(np.mean(track[['Velocity', 'Turning Angle']].diff()**2))
        return np.mean(means, axis=0)

    print('Generating  {} steps from {} steps.\n'.format(
        n_tracks*n_steps, tracks.dropna().__len__()))

    # Generate initial remix
    remix = tracks.dropna()
    remix = remix.ix[np.random.choice(remix.index.values, n_tracks*n_steps)] \
        [['Velocity', 'Turning Angle', 'Rolling Angle']]
    remix['Track_ID'] = 0

    # Shuffle until mean lag is preserved
    ctrl_lags = mean_lags(tracks)*(n_tracks*n_steps-1)
    remix_lags = mean_lags(remix)*(n_tracks*n_steps-1)

    remix = remix.reset_index(drop=True)

    print('Starting at {} total lags, aiming for {}.'.format(
        remix_lags, ctrl_lags))
    iterations = 0
    delta_lags = np.zeros(2)
    diff_lags = remix_lags - ctrl_lags
    while (diff_lags[0] > 0) or (diff_lags[1] > 0):
        index = remix.index.values
        cand = np.random.choice(index[1:-1], 2, replace=False)
        delta_lags[0] = \
            - (remix.ix[cand[0]]['Velocity'] - remix.ix[cand[0]-1]['Velocity'])**2*(cand[0] != cand[1]+1) \
            - (remix.ix[cand[0]]['Velocity'] - remix.ix[cand[0]+1]['Velocity'])**2*(cand[0] != cand[1]-1) \
            - (remix.ix[cand[1]]['Velocity'] - remix.ix[cand[1]-1]['Velocity'])**2*(cand[0] != cand[1]-1) \
            - (remix.ix[cand[1]]['Velocity'] - remix.ix[cand[1]+1]['Velocity'])**2*(cand[0] != cand[1]+1) \
            + (remix.ix[cand[1]]['Velocity'] - remix.ix[cand[0]-1]['Velocity'])**2 \
            + (remix.ix[cand[1]]['Velocity'] - remix.ix[cand[0]+1]['Velocity'])**2 \
            + (remix.ix[cand[0]]['Velocity'] - remix.ix[cand[1]-1]['Velocity'])**2 \
            + (remix.ix[cand[0]]['Velocity'] - remix.ix[cand[1]+1]['Velocity'])**2
        delta_lags[1] = \
            - (remix.ix[cand[0]]['Turning Angle'] - remix.ix[cand[0]-1]['Turning Angle'])**2*(cand[0] != cand[1]+1) \
            - (remix.ix[cand[0]]['Turning Angle'] - remix.ix[cand[0]+1]['Turning Angle'])**2*(cand[0] != cand[1]-1) \
            - (remix.ix[cand[1]]['Turning Angle'] - remix.ix[cand[1]-1]['Turning Angle'])**2*(cand[0] != cand[1]-1) \
            - (remix.ix[cand[1]]['Turning Angle'] - remix.ix[cand[1]+1]['Turning Angle'])**2*(cand[0] != cand[1]+1) \
            + (remix.ix[cand[1]]['Turning Angle'] - remix.ix[cand[0]-1]['Turning Angle'])**2 \
            + (remix.ix[cand[1]]['Turning Angle'] - remix.ix[cand[0]+1]['Turning Angle'])**2 \
            + (remix.ix[cand[0]]['Turning Angle'] - remix.ix[cand[1]-1]['Turning Angle'])**2 \
            + (remix.ix[cand[0]]['Turning Angle'] - remix.ix[cand[1]+1]['Turning Angle'])**2
        if (np.sign(delta_lags[0]) != np.sign(diff_lags[0])) \
            and (np.sign(delta_lags[1]) != np.sign(diff_lags[1])):
            remix_lags += delta_lags
            diff_lags += delta_lags
            index[cand[0]], index[cand[1]] = \
                index[cand[1]], index[cand[0]]
            remix = remix.iloc[index].reset_index(drop=True)
        iterations += 1
        if iterations % 1000 == 0:
            print('  iteration {}, total lag {}.'.format(iterations, remix_lags))

    # print(remix_lags)
    print('Final lags of {} after {} iterations.'.format(
        mean_lags(remix)*(n_tracks*n_steps-1), iterations))

    # Generate new tracks
    new_tracks = pd.DataFrame()
    for i in range(n_tracks):
        track_data = remix.iloc[n_steps*i:n_steps*(i+1)]
        track_data.loc[n_steps*i, 'Rolling Angle'] = np.nan
        new_track = silly_3d_steps(track_data)
        new_track['Track_ID'] = i
        new_tracks = new_tracks.append(new_track)

    if 'Condition' in tracks.columns:
        new_tracks['Condition'] = tracks['Condition'].iloc[0] + ' Remixed preserving lags'
    else:
        new_tracks['Condition'] = 'Remixed preserving lags'

    return new_tracks.reset_index()


if __name__ == '__main__':
    """Test & illustrate rebuilding and remixing tracks"""
    import motility

    tracks = pd.read_csv('Examples/ctrl_data.csv')
    tracks = tracks.drop('index', axis=1)
    ctrl = tracks[tracks.Track_ID == 1015.0]


    """Rebuild a single track"""
    ctrl[['X', 'Y', 'Z']] = ctrl[['X', 'Y', 'Z']] - ctrl[['X', 'Y', 'Z']].iloc[-1]
    rebuilt = silly_3d_steps(ctrl)
    motility.plot_tracks_3d(ctrl.append(rebuilt))
    rebuilt = motility.analyze(rebuilt)
    print(ctrl[['Time', 'Velocity', 'Turning Angle', 'Rolling Angle']])
    print(rebuilt[['Time', 'Velocity', 'Turning Angle', 'Rolling Angle']])


    """Remix Ctrl"""
    # remix = remix(ctrl, n_tracks=1, n_steps=5)
    # remix = motility.analyze(remix)
    # print(remix[['Time', 'Velocity', 'Turning Angle', 'Rolling Angle']])
    # print(ctrl[['Time', 'Velocity', 'Turning Angle', 'Rolling Angle']])


    """Sample dr"""
    sample_dr(tracks)


    """Compare Algorithms"""
    # remidx = remidx(tracks)
    # remix = remix(tracks)
    # remix_lags = remix_preserving_lags(tracks)
    # tracks = tracks.append(remidx)
    # tracks = tracks.append(remix)
    # tracks = tracks.append(remix_lags).reset_index()
    # tracks = motility.analyze(tracks)
    # motility.plot(tracks)
    # motility.lag_plot(tracks, null_model=False)


    """Remix from short vs from long tracks"""
    # summary = motility.summarize(tracks)
    #
    # # Is not prefect, at least if there are non-unique Track_IDs ...
    # short_track_ids = [summary.ix[index]['Track_ID']
    #     for index in summary.sort('Track Duration').index
    #     if summary['Track Duration'].order().cumsum().ix[index]
    #         < summary['Track Duration'].sum()/2]
    #
    # short_remix = remix_preserving_lags(tracks[tracks['Track_ID'].isin(short_track_ids)],
    #     n_tracks=25, n_steps=60)
    # long_remix = remix_preserving_lags(tracks[~tracks['Track_ID'].isin(short_track_ids)],
    #     n_tracks=25, n_steps=60)
    #
    # short_remix['Condition'] = 'Short Tracks Remixed'
    # long_remix['Condition'] = 'Long Tracks Remixed'
    #
    # tracks = tracks.append(short_remix).append(long_remix)
    # tracks = motility.analyze(tracks.reset_index())
    # motility.plot(tracks)


    """Create long tracks"""
    # remix = remix_preserving_lags(tracks, n_tracks=5, n_steps=8*60*3)
    # remix.to_csv('long_remix.csv')
