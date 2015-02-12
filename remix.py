"""Remix cell tracks"""
import numpy as np
import pandas as pd


def silly_3d_steps(n_steps=10):
    """Generate a 3D random walk"""
    # velocities = np.cumsum(np.ones(n_steps))
    # turning_angles = np.zeros(n_steps-1)
    velocities = np.random.lognormal(0, 0.5, n_steps)
    turning_angles = np.random.lognormal(0, 0.5, n_steps-1)
    rolling_angles = (np.random.rand(n_steps-2) - 0.5)*2*np.pi

    # Walk in x-y plane w/ given velocity and turning angles
    dr = np.zeros((n_steps+1, 3))
    dr[1,0] = velocities[0]
    for i in range(2, n_steps+1):
        cosa = np.cos(turning_angles[i-2])
        sina = np.sin(turning_angles[i-2])
        dr[i,0] = (cosa*dr[i-1,0] - sina*dr[i-1,1])*velocities[i-1]/velocities[i-2]
        dr[i,1] = (sina*dr[i-1,0] + cosa*dr[i-1,1])*velocities[i-1]/velocities[i-2]

    # Add z = 0 and move 3rd position to origin
    r = np.cumsum(dr, axis=0) - dr[1,:] - dr[2,:]

    # Rotate moved positions minus the rolling angles around the next step
    for i in range(3, n_steps+1):
        cost = np.cos(-rolling_angles[i-3])
        sint = np.sin(-rolling_angles[i-3])
        n_vec = dr[i,:]/np.sqrt(np.sum(dr[i,:]*dr[i,:]))
        for j in range(i):
            cross_prod = np.cross(n_vec, r[j,:])
            dot_prod = np.sum(n_vec*r[j,:])
            r[j,:] = r[j,:]*cost + cross_prod*sint + n_vec*dot_prod*(1 - cost)
        r = r - dr[i,:]

    return pd.DataFrame({'X': r[:,0], 'Y': r[:,1], 'Z': r[:,2]})


if __name__ == '__main__':
    """Test & illustrate module"""
    import lana


    """Rebuild a single track"""
    tracks = pd.read_csv('Examples/ctrl_data.csv')
    track = silly_3d_steps()
    lana.plot_tracks_3d(track)


    """Remix data"""
    # track = silly_3d_steps(1000)
    # track['Track_ID'] = 1
    # track = lana.analyze_motility(track)
    # lana.plot_motility(track)
