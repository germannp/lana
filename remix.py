"""Remix cell tracks"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def silly_3d_steps(track_data=None, n_steps=10):
    """Generate a walk from track data (i.e. velocities, turning & rolling angles)"""
    if type(track_data) != pd.core.frame.DataFrame:
        print('No track data given, using random motility parameters.')
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

    # Add z = 0 and move 2nd position to origin
    r = np.cumsum(dr, axis=0) - dr[1,:] - dr[2,:]

    # Rotate moved positions minus the rolling angles around the next step
    for i in range(3, n_steps+1):
        cost = np.cos(-rolling_angles[i-3])
        sint = np.sin(-rolling_angles[i-3])
        n_vec = dr[i-1,:]/np.sqrt(np.sum(dr[i-1,:]*dr[i-1,:]))
        for j in range(i-1):
            cross_prod = np.cross(n_vec, r[j,:])
            dot_prod = np.sum(n_vec*r[j,:])
            r[j,:] = r[j,:]*cost + cross_prod*sint + n_vec*dot_prod*(1 - cost)
        r = r - dr[i,:]

    return pd.DataFrame({'Time': np.arange(n_steps+1), 'X': -r[:,0], 'Y': -r[:,1],
        'Z': -r[:,2], 'Source': 'Silly 3D walk', 'Condition': condition})


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


def remix_preserving_lag(tracks, n_tracks=50, n_steps=60):
    """Return new tracks generating by remixing while preserving velocity lag"""
    def mean_lag(tracks):
        """Calculate mean lag in velocity of track(s)"""
        criteria = [crit for crit in ['Condition', 'Sample', 'Track_ID']
            if crit in tracks.columns]
        means = []
        for _, track in tracks.groupby(criteria):
            means.append(track['Velocity'].diff().abs().mean())
        return np.mean(means)

    print('Generating  {} steps from {} steps.\n'.format(
        n_tracks*n_steps, tracks.dropna().__len__()))

    # Generate initial remix
    remix = tracks.dropna()
    remix = remix.ix[np.random.choice(remix.index.values, n_tracks*n_steps)] \
        [['Velocity', 'Turning Angle', 'Rolling Angle']]
    remix['Track_ID'] = 0

    # Shuffle until mean lag is preserved
    ctrl_lag = mean_lag(tracks)*(n_tracks*n_steps-1)
    remix_lag = mean_lag(remix)*(n_tracks*n_steps-1)

    remix = remix.reset_index(drop=True)

    iterations = 0
    print('Starting at {} total lag, aiming for {}.'.format(remix_lag, ctrl_lag))
    while remix_lag > ctrl_lag:
        index = remix.index.values
        cand = np.random.choice(index[1:-1], 2, replace=False)
        delta_lag = \
            - abs(remix.ix[cand[0]]['Velocity'] - remix.ix[cand[0]-1]['Velocity'])*(cand[0] != cand[1]+1) \
            - abs(remix.ix[cand[0]]['Velocity'] - remix.ix[cand[0]+1]['Velocity'])*(cand[0] != cand[1]-1) \
            - abs(remix.ix[cand[1]]['Velocity'] - remix.ix[cand[1]-1]['Velocity'])*(cand[0] != cand[1]-1) \
            - abs(remix.ix[cand[1]]['Velocity'] - remix.ix[cand[1]+1]['Velocity'])*(cand[0] != cand[1]+1) \
            + abs(remix.ix[cand[1]]['Velocity'] - remix.ix[cand[0]-1]['Velocity']) \
            + abs(remix.ix[cand[1]]['Velocity'] - remix.ix[cand[0]+1]['Velocity']) \
            + abs(remix.ix[cand[0]]['Velocity'] - remix.ix[cand[1]-1]['Velocity']) \
            + abs(remix.ix[cand[0]]['Velocity'] - remix.ix[cand[1]+1]['Velocity'])
        if delta_lag < 0:
            remix_lag += delta_lag
            index[cand[0]], index[cand[1]] = \
                index[cand[1]], index[cand[0]]
            remix = remix.iloc[index].reset_index(drop=True)
        iterations += 1
        if iterations % 1000 == 0:
            print('  iteration {}, total lag {}.'.format(iterations, remix_lag))

    print('  Final lag of {} after {} iterations.'.format(
        mean_lag(remix)*(n_tracks*n_steps-1), iterations))

    new_tracks = pd.DataFrame()
    for i in range(n_tracks):
        track_data = remix.iloc[n_steps*i:n_steps*(i+1)]
        track_data.loc[n_steps*i, 'Rolling Angle'] = np.nan
        new_track = silly_3d_steps(track_data)
        new_track['Track_ID'] = i
        new_tracks = new_tracks.append(new_track)

    if 'Condition' in tracks.columns:
        new_tracks['Condition'] = tracks['Condition'].iloc[0] + ' Remixed preserving lag'
    else:
        new_tracks['Condition'] = 'Remixed preserving lag'

    return new_tracks.reset_index()


def remix_speed_ratio(tracks, n_tracks=50, n_steps=60):
    """Remix tracks according to speed ratio rather than speed"""
    """Return new tracks generated by remixing given tracks"""
    print('Generating  {} steps from {} steps.'.format(
        n_tracks*n_steps, tracks.__len__()))

    velo_and_turn = tracks[tracks['Rolling Angle'].isnull()] \
        [['Velocity', 'Turning Angle']].dropna()

    criteria = [crit for crit in ['Condition', 'Sample', 'Track_ID']
        if crit in tracks.columns]
    ratios_and_angles = pd.DataFrame()
    for _, track in tracks.groupby(criteria):
        ratios = track[1:]['Velocity'].values/track[:-1]['Velocity'].values
        ratios = np.append(ratios, np.nan)
        ratios_and_angles = ratios_and_angles.append(pd.DataFrame({
            'Velocity': ratios,
            'Turning Angle': track['Turning Angle'],
            'Rolling Angle': track['Rolling Angle']}))
    ratios_and_angles = ratios_and_angles.dropna().reset_index()

    new_tracks = pd.DataFrame()
    for track_id in range(n_tracks):
        track_data = velo_and_turn.ix[np.random.choice(velo_and_turn.index.values, 1)]
        track_data = track_data.append(
            ratios_and_angles.ix[np.random.choice(ratios_and_angles.index.values, n_steps-1)])
        track_data['Velocity'] = track_data['Velocity'].cumprod()
        new_track = silly_3d_steps(track_data)
        new_track['Track_ID'] = track_id
        new_tracks = new_tracks.append(new_track)

    if 'Condition' in tracks.columns:
        new_tracks['Condition'] = tracks['Condition'].iloc[0] + ' Remixed by Velo-Ratio'
    else:
        new_tracks['Condition'] = 'Remixed by Velo-Ratio'

    return new_tracks.reset_index()


def sample_given_last_step(tracks, n_tracks=50, n_steps=60):
    """Returns new tracks generated by sampling given tracks"""
    initial_data = tracks.dropna()
    print('Generating  {} steps from {} steps.'.format(
        n_tracks*n_steps, initial_data.__len__()))

    # Learn conditional, multivariant KDE
    dep_data = initial_data[initial_data['Track Time'] != 2]
    criteria = [crit for crit in ['Condition', 'Sample', 'Track_ID']
        if crit in initial_data.columns]
    indep_data = pd.DataFrame()
    for _, track in initial_data.groupby(criteria):
        indep_data = indep_data.append(track.iloc[:-1])
    next_step_kde = sm.nonparametric.KDEMultivariateConditional(
        dep_data[['Velocity', 'Turning Angle', 'Rolling Angle']].values,
        indep_data[['Velocity', 'Turning Angle', 'Rolling Angle']].values,
        dep_type='ccc', indep_type='ccc', bw='normal_reference')
    max_kde = max(next_step_kde.pdf())

    # Generate new tracks
    new_tracks = pd.DataFrame()
    for track_id in range(n_tracks):
        track = initial_data.ix[np.random.choice(initial_data.index.values, 1)] \
            [['Velocity', 'Turning Angle', 'Rolling Angle']]
        max_index = max(track.index)
        while track.__len__() < n_steps+1:
            candidate = [
                np.random.rand()*np.percentile(initial_data['Velocity'], 99.5),
                np.random.rand()*np.pi,
                (np.random.rand() - 0.5)*2*np.pi]
            r = np.random.rand()
            if next_step_kde.pdf(candidate, track.iloc[-1]) > max_kde*r:
                max_index = max_index+1
                track = track.append(pd.DataFrame({'Velocity': candidate[0],
                    'Turning Angle': candidate[1],
                    'Rolling Angle': candidate[2]}, index=[max_index]))
        track.loc[max_index-n_steps, 'Rolling Angle'] = np.nan
        new_track = silly_3d_steps(track)
        new_track['Track_ID'] = track_id
        new_tracks = new_tracks.append(new_track)

    if 'Condition' in tracks.columns:
        new_tracks['Condition'] = tracks['Condition'].iloc[0] + ' Sampled'
    else:
        new_tracks['Condition'] = 'Sampled'

    return new_tracks.reset_index()


def sample_roll_indep(tracks, n_tracks=50, n_steps=60):
    """Sample Rolling Angle independently"""
    print('Generating  {} steps from {} steps.'.format(
        n_tracks*n_steps, tracks.__len__()))

    # Learn conditional, multivariant KDE of velocities & turning angles
    initial_data = tracks.dropna()
    dep_data = initial_data[initial_data['Track Time'] != 2]
    criteria = [crit for crit in ['Condition', 'Sample', 'Track_ID']
        if crit in initial_data.columns]
    indep_data = pd.DataFrame()
    for _, track in initial_data.groupby(criteria):
        indep_data = indep_data.append(track.iloc[:-1])
    next_step_kde = sm.nonparametric.KDEMultivariateConditional(
        dep_data[['Velocity', 'Turning Angle']].values,
        indep_data[['Velocity', 'Turning Angle']].values,
        dep_type='cc', indep_type='cc', bw='normal_reference')
    max_kde = max(next_step_kde.pdf())

    # Learn Rolling Angles
    rolling_angles_kde = sm.nonparametric.KDEUnivariate(
        tracks['Rolling Angle'].dropna())
    rolling_angles_kde.fit()
    max_rolling_kde = max(rolling_angles_kde.density)

    # Generate velocities and turning angles of the new tracks
    new_tracks = pd.DataFrame()
    for track_id in range(n_tracks):
        track = initial_data.ix[np.random.choice(initial_data.index.values, 1)] \
            [['Velocity', 'Turning Angle']]
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
            if track.__len__() <= 1:
                rolled = True
            else:
                rolled = False
            while not rolled:
                candidate_rolling_angle = (np.random.rand() - 0.5)*2*np.pi
                r = np.random.rand()*max_rolling_kde
                if rolling_angles_kde.evaluate(candidate_rolling_angle) > r:
                    track.loc[max_index, 'Rolling Angle'] = candidate_rolling_angle
                    rolled = True
        new_track = silly_3d_steps(track)
        new_track['Track_ID'] = track_id
        new_tracks = new_tracks.append(new_track)

    if 'Condition' in tracks.columns:
        new_tracks['Condition'] = tracks['Condition'].iloc[0] + ' Sampled'
    else:
        new_tracks['Condition'] = 'Sampled'

    return new_tracks.reset_index()


def sample_remix(tracks, n_tracks=50, n_steps=60):
    """Returns remixed tracks selected for previous velocity"""
    initial_data = tracks.dropna()
    print('Generating  {} steps from {} steps.'.format(
        n_tracks*n_steps, initial_data.__len__()))

    # Learn conditional, multivariant KDE
    dep_data = initial_data[initial_data['Track Time'] != 2]
    criteria = [crit for crit in ['Condition', 'Sample', 'Track_ID']
        if crit in initial_data.columns]
    indep_data = pd.DataFrame()
    for _, track in initial_data.groupby(criteria):
        indep_data = indep_data.append(track.iloc[:-1])
    next_step_kde = sm.nonparametric.KDEMultivariateConditional(
        dep_data['Velocity'].values,
        indep_data['Velocity'].values,
        dep_type='c', indep_type='c', bw='normal_reference')
    max_kde = max(next_step_kde.pdf())

    # Generate new tracks
    new_tracks = pd.DataFrame()
    for track_id in range(n_tracks):
        track = initial_data.ix[np.random.choice(initial_data.index.values, 1)] \
            [['Velocity', 'Turning Angle', 'Rolling Angle']]
        while track.__len__() < n_steps+1:
            candidate = initial_data.ix[np.random.choice(initial_data.index.values, 1)] \
                [['Velocity', 'Turning Angle', 'Rolling Angle']]
            r = np.random.rand()
            if next_step_kde.pdf(candidate['Velocity'],
                [track.iloc[-1]['Velocity']]) > max_kde*r:
                track = track.append(candidate)
        # track.loc[max_index-n_steps, 'Rolling Angle'] = np.nan
        new_track = silly_3d_steps(track)
        new_track['Track_ID'] = track_id
        new_tracks = new_tracks.append(new_track)

    if 'Condition' in tracks.columns:
        new_tracks['Condition'] = tracks['Condition'].iloc[0] + ' Sampled Remix'
    else:
        new_tracks['Condition'] = 'Sampled Remix'

    return new_tracks.reset_index()


if __name__ == '__main__':
    """Test & illustrate rebuilding and remixing tracks"""
    import lana
    tracks = pd.read_csv('Examples/ctrl_data.csv')
    tracks = tracks.drop('index', axis=1)
    ctrl = tracks[tracks.Track_ID == 1015.0]


    """Rebuild a single track"""
    # ctrl[['X', 'Y', 'Z']] = ctrl[['X', 'Y', 'Z']] - ctrl[['X', 'Y', 'Z']].iloc[-1]
    # rebuilt = silly_3d_steps(ctrl)
    # # lana.plot_tracks_3d(ctrl.append(rebuilt)) # TODO: Nice rotation ...
    # rebuilt = lana.analyze_motility(rebuilt)
    # print(ctrl[['Time', 'Velocity', 'Turning Angle', 'Rolling Angle']])
    # print(rebuilt[['Time', 'Velocity', 'Turning Angle', 'Rolling Angle']])


    """Remix Ctrl"""
    # remix = remix(ctrl, n_tracks=1, n_steps=5)
    # remix = lana.analyze_motility(remix)
    # print(remix[['Time', 'Velocity', 'Turning Angle', 'Rolling Angle']])
    # print(ctrl[['Time', 'Velocity', 'Turning Angle', 'Rolling Angle']])


    """Remix tracks from koordinates (Gerard et al. 2014)"""
    # remidx = remidx(tracks)
    # remix = remix(tracks)
    # tracks = tracks.append(remidx)
    # tracks = tracks.append(remix).reset_index()
    # tracks = lana.analyze_motility(tracks)
    # lana.plot_motility(tracks)


    """Remix tracks from motility parameters"""
    # remix = remix(tracks)
    # tracks = tracks.append(remix).reset_index()
    # tracks = lana.analyze_motility(tracks)
    # lana.plot_motility(tracks)
    # lana.lag_plot(tracks)
    # lana.plot_joint_motility(tracks[tracks.Condition == 'Ctrl Remixed'])


    """Remix from short vs from long tracks"""
    # summary = lana.summarize_tracks(tracks)
    #
    # # Is not prefect, at least if there are non-unique Track_IDs ...
    # short_track_ids = [summary.ix[index]['Track_ID']
    #     for index in summary.sort('Track Duration').index
    #     if summary['Track Duration'].order().cumsum().ix[index]
    #         < summary['Track Duration'].sum()/2]
    #
    # short_remix = remix(tracks[tracks['Track_ID'].isin(short_track_ids)])
    # long_remix = remix(tracks[~tracks['Track_ID'].isin(short_track_ids)])
    #
    # short_remix['Condition'] = 'Short Tracks Remixed'
    # long_remix['Condition'] = 'Long Tracks Remixed'
    #
    # tracks = tracks.append(short_remix).append(long_remix)
    # tracks = lana.analyze_motility(tracks.reset_index())
    # lana.plot_motility(tracks)


    """Sample from KDE given last step"""
    # sample = sample_given_last_step(tracks)
    # tracks = tracks.append(sample).reset_index()
    # tracks = lana.analyze_motility(tracks)
    # lana.plot_motility(tracks)
    # lana.lag_plot(tracks)
    # lana.plot_joint_motility(tracks)


    """Sample Rolling Angle independently"""
    # sample = sample_roll_indep(tracks)
    # tracks = tracks.append(sample).reset_index()
    # tracks = lana.analyze_motility(tracks)
    # lana.plot_motility(tracks)
    # lana.lag_plot(tracks)
    # lana.plot_joint_motility(tracks)


    """Remix with rejection based on velocity"""
    # sample = sample_remix(tracks)
    # tracks = tracks.append(sample).reset_index()
    # tracks = lana.analyze_motility(tracks)
    # lana.plot_motility(tracks)
    # lana.lag_plot(tracks)
    # lana.plot_joint_motility(tracks)


    """Remix based on ratio of consecutive velocities"""
    # remix = remix_speed_ratio(tracks)
    # tracks = tracks.append(remix).reset_index()
    # tracks = lana.analyze_motility(tracks)
    # lana.plot_motility(tracks)


    """Remix with preserved lag"""
    remix_lag = remix_preserving_lag(tracks)
    remix = remix(tracks)
    tracks = tracks.append(remix_lag)
    tracks = tracks.append(remix).reset_index()
    tracks = lana.analyze_motility(tracks)
    lana.plot_motility(tracks)
    lana.lag_plot(tracks)
