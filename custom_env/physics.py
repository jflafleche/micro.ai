"""
Simple 2D physics simulator.
"""

import numpy as np
import math
from scipy.spatial.distance import pdist, cdist, squareform

class PhysicsEnvironment():
    def __init__(
        self,
        N,
        particle_mass,
        agent_mass,
        particle_size,
        agent_size,
        bounds,
        fluid_viscosity,
        fluid_density,
        interval_dt=0.01
    ):
        self.N = N
        self.particle_mass = particle_mass
        self.agent_mass = agent_mass
        self.particle_size = particle_size
        self.agent_size = agent_size
        self.interval_dt = interval_dt
        self.bounds = bounds
        self.fluid_viscosity = fluid_viscosity
        self.fluid_density = fluid_density

        self._gen_states()

    def _gen_states(self):
        self.state = np.zeros((self.N,4))
        n = 0
        while n < self.N:
            # generate a random position and velocity
            pos = np.random.random((2))
            # scale to bounds and ensure away from wall
            pos[0] *= (self.bounds[1] - self.bounds[0])*0.8
            pos[1] *= (self.bounds[3] - self.bounds[2])*0.8
            pos[:2] += 1

            if n == 0:
                size = self.agent_size
            else:
                size = self.particle_size

            overlap = False
            for i in range(0,self.N):
                d = pdist((pos, self.state[i,:2]))
                if d < size * 2:
                    overlap = True
            if not overlap:
                self.state[n,:2] = pos
                n += 1
        
        self.sizes = np.ones((self.N)) * self.particle_size
        self.sizes[0] = self.agent_size
        self.masses = np.ones((self.N)) * self.particle_mass
        self.masses[0] = self.agent_mass

    def _dist(self, state):
        pd = pdist(state)
        spd = squareform(pd)
        spd = spd - self.sizes - self.sizes[1]
        return spd

    def _resolve_collisions(self):
        done = False
        particle_touch = []
        # update positions by adding velocities * dt
        self.state[:, :2] += self.state[:, 2:] * self.interval_dt

        # find pairs of particles that will likely undergo a collision
        # next interval

        predicted = self.state[:, :2] + self.state[:, 2:] * self.interval_dt
        dists = self._dist(predicted)
        agent_dists = dists[0,:]
        ind1, ind2 = np.where(dists < 0)
        unique = (ind1 > ind2)
        ind1 = ind1[unique]
        ind2 = ind2[unique]

        # update velocities of colliding pairs
        # ref. https://github.com/zjost/pycollide/blob/master/pycollide/pycollide.py
        # ref http://vobarian.com/collisions/2dcollisions2.pdf
        for i1, i2 in zip(ind1, ind2):
            # mass
            m1 = self.masses[i1]
            m2 = self.masses[i2]

            # location vectors
            r1 = self.state[i1, :2]
            r2 = self.state[i2, :2]
            d_o = pdist((r1, r2))
            # velocity vectors
            v1 = self.state[i1, 2:]
            v2 = self.state[i2, 2:]

            # normal vector
            n_hat_p = r2 - r1
            n_hat = n_hat_p / np.linalg.norm(n_hat_p)

            # tangent vector
            t_hat = [-n_hat[1], n_hat[0]]

            # projected velocity vectors
            v1n = np.dot(v1, n_hat)
            v1t = np.dot(v1, t_hat)
            v2n = np.dot(v2, n_hat)
            v2t = np.dot(v2, t_hat)

            v1n_p = (v1n*(m1 - m2) + 2*m2*v2n) / (m1 + m2)
            v2n_p = (v2n*(m2 - m1) + 2*m1*v1n) / (m1 + m2)

            v1_p = np.array([v1n_p, v1t])
            v2_p = np.array([v2n_p, v2t])

            # Define conversion matrix
            A11 = np.dot(np.array([1, 0]), n_hat)
            A12 = np.dot(np.array([1, 0]), t_hat)
            A21 = np.dot(np.array([0, 1]), n_hat)
            A22 = np.dot(np.array([0, 1]), t_hat)
            A = np.array([[A11, A12], [A21, A22]])

            v1_new = np.dot(A, v1_p)
            v2_new = np.dot(A, v2_p)

            predicted1 = r1 + self.interval_dt * v1_new
            predicted2 = r2 + self.interval_dt * v2_new

            d_p = pdist((predicted1, predicted2))

            c1 = d_p
            c2 = (self.sizes[i1] + self.sizes[i2]) - d_p          

            x = -float((r1[0]-r2[0])*c2/c1)
            y = -float((r1[1]-r2[1])*c2/c1)
            
            self.state[i1, 2:] = v1_new
            self.state[i2, 2:] = v2_new

            # bring apart particles that are stuck in overlap
            i = 0
            while d_p < self.sizes[1]*2 and i < 1000:
                self.state[i2,:2] += [x,y]
                predicted1 = r1 + self.interval_dt * v1_new
                predicted2 = r2 + self.interval_dt * v2_new
                d_p = pdist((predicted1, predicted2))
                i += 1
            if i >= 1000:
                # if still stuck, reset environment
                done = True

        # check for crossing boundary
        crossed_x1 = (self.state[:, 0] < self.bounds[0] + self.sizes)
        crossed_x2 = (self.state[:, 0] > self.bounds[1] - self.sizes)
        crossed_y1 = (self.state[:, 1] < self.bounds[2] + self.sizes)
        crossed_y2 = (self.state[:, 1] > self.bounds[3] - self.sizes)

        # check if particle touched a wall
        wall_touch = any(crossed_y1) or any(crossed_y2)

        self.state[crossed_x1, 0] = self.bounds[0] + self.sizes[crossed_x1]
        self.state[crossed_x2, 0] = self.bounds[1] - self.sizes[crossed_x2]

        self.state[crossed_y1, 1] = self.bounds[2] + self.sizes[crossed_y1]
        self.state[crossed_y2, 1] = self.bounds[3] - self.sizes[crossed_y2]

        self.state[crossed_x1 | crossed_x2, 2] *= -1
        self.state[crossed_y1 | crossed_y2, 3] *= -1

        return done, agent_dists, wall_touch

    def _resolve_drag(self):
        # ref (2.27) from Micro-Scale Mobile Robotics - Eric Diller and Metin Sitti
        # approximation for a sphere at low Reynolds number
        F_dx = self.sizes * 6 * math.pi * self.fluid_viscosity * self.state[:,2]
        F_dy = self.sizes * 6 * math.pi * self.fluid_viscosity * self.state[:,3]
        self.state[:,2] -= F_dx / self.masses * self.interval_dt
        self.state[:,3] -= F_dy / self.masses * self.interval_dt

    def _resolve_forces(self, Fx, Fy):
        # magnetic forces applied to agent
        self.state[0, 2] += Fx / self.masses[0] * self.interval_dt
        self.state[0, 3] += Fy / self.masses[0] * self.interval_dt

    def step(self, Fx, Fy):
        # take care of particle and wall collisions
        done, agent_dists, wall_touch = self._resolve_collisions()
        
        # add force
        self._resolve_forces(Fx, Fy)

        # add drag
        self._resolve_drag()

        return self.state, done, agent_dists, wall_touch

    def reset(self):
        self._gen_states()
        dists = self._dist(self.state)
        return self.state, dists[0, :]
