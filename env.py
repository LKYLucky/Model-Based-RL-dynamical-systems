import numpy as np
from scipy.integrate import odeint
import gym


# we could generalize/modify this class to take a more generic ODE function as input, or generalise self.f(...)

class ODEBaseEnv(gym.Env):

    def __init__(self, num_species=2, time_interval_action=1, dt=1e-3, init_state=[], init_species=[], init_f = [],
                 rate_constants=[]):
        # may need to add more here

        low = np.zeros((num_species), dtype=np.float32)
        high = np.array([np.finfo(np.float32).max] * num_species, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(num_species + 1)
        # self.action_space = gym.spaces.Box(low, high, dtype=np.float32) ##replace with gym.spaces.discrete
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

        self.N = num_species
        self.tau = time_interval_action  # we'd need to modify this if we consider irregular time intervals between observations and/or actions
        self.dt = dt

        self.init_state = init_state
        self.init_species = init_species  # add this
        self.init_f = init_f #add this
        self.rate_constants = rate_constants
        # self.species = species

    def step(self, action):
        'integrates the ODE system with a constant additive force vector (action)'
        'returns a reward based on the ODE variables at the *end* of this time interval'

        tt = np.linspace(0, self.tau, int(1 / self.dt))
        state_prev = self.state
        self.u = 0.1 * action
        z_init = self.state
        z = odeint(self.f, z_init, tt)
        # print("z",z)
        self.state = z[-1]
        state_curr = self.state
        # print("state_prev", state_prev, "state_curr",  state_curr)
        reward = - (((state_prev - state_curr) / (state_prev + state_curr)) ** 2).sum()
        reward *= 1e3
        done = False  # indicates whether the episode is terminated; optional
        info = {}  # can be used in custom Gym environments; optional

        # we assume z is observed with zero noise
        obs = self.state
        return obs, reward, done, info, z

    def reset(self):
        'resets the ODE system to initial conditions'
        self.state = self.init_state
        self.species = self.init_species
        self.f_const = self.init_f
        # print("state", self.state)
        return self.state, self.species, self.f_const


class LotkaVolterraEnv(ODEBaseEnv):

    def f(self, Z, t):
        # self.rate_constants = [0.1, 0.05, 0.05]
        k1 = self.rate_constants[0]
        k2 = self.rate_constants[1]
        k3 = self.rate_constants[2]

        X, Y = Z
        Zdot = [k1 * X - k2 * X * Y, k2 * X * Y - k3 * Y]

        Zdot += self.u
        return Zdot


class BrusselatorEnv(ODEBaseEnv):

    def f(self, Z, t):
        # A = 1
        # B = 3
        # self.species = [1, 1.7]
        # self.species = [1, 3] #unstable
        A = self.species[0]
        B = self.species[1]

        k1 = self.rate_constants[0]
        k2 = self.rate_constants[1]
        k3 = self.rate_constants[2]
        k4 = self.rate_constants[3]

        X, Y = Z
        Zdot = [k1 * A + k2 * X ** 2 * Y - k3 * B * X - k4 * X, - k2 * X ** 2 * Y + k3 * B * X]

        return Zdot

    def step(self, action):
        'integrates the ODE system with a constant additive force vector (action)'
        'returns a reward based on the ODE variables at the *end* of this time interval'

        tt = np.linspace(0, self.tau, int(1 / self.dt))
        state_prev = self.state
        sc_prev = self.species
        self.u =  0.1* action

        self.species = sc_prev + self.u

        z_init = self.state
        z = odeint(self.f, z_init, tt)
        self.state = z[-1]
        state_curr = self.state

        #reward = -((state_prev - state_curr) ** 2).sum() #change reward function back to original
        reward = -(((state_prev - state_curr) / (state_prev + state_curr)) ** 2).sum()
        reward *= 10
        #reward *= 1e3
        # print("reward", reward)
        done = False  # indicates whether the episode is terminated; optional
        info = {}  # can be used in custom Gym environments; optional

        # we assume z is observed with zero noise

        # add minimum constraints for A and B
        #self.species[0] = max(self.species[0], 0.1)
        #self.species[1] = max(self.species[1], 0.1)

        obs = self.species
        return obs, reward, done, info, z


class GeneralizedEnv(ODEBaseEnv):

    def f(self, Z, t):
        # Zdot =  LotkaVolterraEnv.Zdot + BrusselatorEnv.Zdot

        self.species = [1, 1.7]  # unstable
        A = self.species[0]
        B = self.species[1]

        k1 = self.rate_constants[0]
        k2 = self.rate_constants[1]
        k3 = self.rate_constants[2]
        k4 = self.rate_constants[3]
        k5 = self.rate_constants[4]
        k6 = self.rate_constants[5]

        X, Y = Z
        dX = k1 * X - k2 * X * Y + k4 * A + k5 * X ** 2 * Y - k6 * B * X
        dY = k2 * X * Y - k3 * Y - k5 * X ** 2 * Y + k6 * B * X
        Zdot = [dX, dY]
        Zdot += self.u
        return Zdot


class OregonatorEnv(ODEBaseEnv):
    def f(self, S, t):
        #self.species = [0.06, 0.02]
        #self.species = [0.2, 1]
        #self.species = [2, 1]
        self.species = [3, 0.02]
        A = self.species[0]
        B = self.species[1]

        k1 = self.rate_constants[0]
        k2 = self.rate_constants[1]
        k3 = self.rate_constants[2]
        k4 = self.rate_constants[3]
        k5 = self.rate_constants[4]

        f = self.f_const[0]


        #f = 10.538
        X, Y, Z = S

        dX = k1 * A * Y - k2 * X * Y + k3 * A * X - 2 * k4 * X ** 2
        dY = -k1 * A * Y - k2 * X * Y + 0.5 * f * k5 * B * Z
        dZ = 2 * k3 * A * X - k5 * B * Z
        Sdot = [dX, dY, dZ]
        Sdot += self.u
        return Sdot

    def step(self, action):
        'integrates the ODE system with a constant additive force vector (action)'
        'returns a reward based on the ODE variables at the *end* of this time interval'

        tt = np.linspace(0, self.tau, int(1 / self.dt))
        state_prev = self.state
        f_prev = self.f_const
        self.u = 0.5 * action
        self.f_const = f_prev + self.u

        z_init = self.state
        z = odeint(self.f, z_init, tt)
        self.state = z[-1]
        state_curr = self.state
        reward = - (((state_prev - state_curr) / (state_prev + state_curr)) ** 2).sum()
        done = False  # indicates whether the episode is terminated; optional
        info = {}  # can be used in custom Gym environments; optional

        # we assume z is observed with zero noise
        obs = self.f_const
        return obs, reward, done, info, z