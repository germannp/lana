"""Remix cell tracks"""
import numpy as np
import pandas as pd


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
        n_steps = track_data.__len__()
        velocities = track_data['Velocity'].dropna().values
        turning_angles = track_data['Turning Angle'].dropna().values
        rolling_angles = track_data['Rolling Angle'].dropna().values
        if 'Condtion' in track_data.columns:
            condition = track_data['Condition'].iloc[0] + ' Rebuilt'
        else:
            condition = 'Rebuilt'

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


def remix(tracks=None, n_tracks=50, n_steps=60):
    """Generate new tracks by remixing motility parameters from given tracks"""
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
            remaining_data.ix[np.random.choice(remaining_data.index.values, n_steps-3)])
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


if __name__ == '__main__':
    """Test & illustrate rebuilding and remixing tracks"""
    import lana


    """Rebuild a single track"""
    tracks = pd.read_csv('Examples/ctrl_data.csv')
    # ctrl = tracks[tracks.Track_ID == 1015.0]
    # ctrl[['X', 'Y', 'Z']] = ctrl[['X', 'Y', 'Z']] - ctrl[['X', 'Y', 'Z']].iloc[-1]
    # rebuilt = silly_3d_steps(ctrl)
    # lana.plot_tracks_3d(ctrl.append(rebuilt)) # TODO: Nice rotation ...
    # rebuilt = lana.analyze_motility(rebuilt)
    # print(ctrl[['Time', 'Velocity', 'Turning Angle', 'Rolling Angle']])
    # print(rebuilt[['Time', 'Velocity', 'Turning Angle', 'Rolling Angle']])


    """Remix Ctrl"""
    # remix = remix(ctrl, n_tracks=1, n_steps=6)
    # remix = lana.analyze_motility(remix)
    # print(remix[['Time', 'Velocity', 'Turning Angle', 'Rolling Angle']])
    # print(ctrl[['Time', 'Velocity', 'Turning Angle', 'Rolling Angle']])


    """Remix tracks"""
    remix = remix(tracks)
    tracks = tracks.append(remix).reset_index()
    tracks = lana.analyze_motility(tracks)
    lana.plot_motility(tracks)
    lana.lag_plot(tracks)
    lana.plot_joint_motility(tracks[tracks.Condition == 'Ctrl Remixed'])
