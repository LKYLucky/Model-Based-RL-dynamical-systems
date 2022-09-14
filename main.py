import numpy as np
from matplotlib import pyplot as plt
from env import LotkaVolterraEnv, BrusselatorEnv, GeneralizedEnv, OregonatorEnv
from env_model import LotkaVolterraEnvModel, BrusselatorEnvModel, GeneralizedEnvModel, OregonatorEnvModel

import torch
import scipy.optimize as so
from model import RateConstantModel

# import torch.nn.functional as F
# from stable_baselines.sac.policies import MlpPolicy
# from stable_baselines import SAC
from mpl_toolkits.mplot3d import Axes3D


class Critic(torch.nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.net(state)


def get_z_init(ODE_env):
    if ODE_env == "Oregonator":
        Z_init = np.array([1.0, 1.0, 1.0])
    else:
        prey = np.random.uniform(0.5, 1.5)
        pred = np.random.uniform(1, 3)
        Z_init = np.array([prey, pred])

    # Z_init = np.array([1,1])
    return Z_init


def uphill_policy(observation, critic):
    state = torch.tensor(observation, dtype=torch.float, requires_grad=True)
    critic(state).backward()
    # print("critic(state)",critic(state))
    u = state.grad.detach().numpy() * 0.1
    return u


def discounted_rewards(rewards, R, gamma):
    returns = np.zeros_like(rewards)
    ### remove last 25 time steps
    for t in reversed(range(len(rewards))):
        R = R * gamma + rewards[t]
        returns[t] = R
    return returns


def update_policy(states, returns, critic, critic_optimizer):
    states = torch.tensor(states, dtype=torch.float)
    returns = torch.tensor(returns, dtype=torch.float)

    values = critic(states)

    advantages = returns - torch.reshape(values, (-1,))
    # loss for critic (MSE)

    critic_loss = advantages.pow(2).mean()

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()


def find_rate_constants(Z_arr, theta_arr, rc_model, action):
    result = rc_model.solve_minimize(Z_arr, theta_arr, dt, action)
    rc_model.rates = result.x
    return result


def run_one_episode(env_option, max_step, critic, Z_arr, theta_arr, rc_model, calc_rate, n_episode):
    states = []
    rewards = []

    Z_init = get_z_init(ODE_env)
    env_option.init_state = Z_init
    if ODE_env == "Brusselator":

        #env_option.init_species = np.array([1.0, 3.0])
        A = np.random.uniform(0.8, 1.2)
        B = np.random.uniform(2.6, 3.4)
        env_option.init_species = np.array([A, B])


    elif ODE_env == "Oregonator":
        #env_option.init_species = np.array([2, 1, 0])
        #env_option.init_f = np.array([1.9475, 0, 0])
        #env_option.init_f = np.array([1, 0, 0])
        env_option.init_f = np.array([1, 0, 0])


    Z_history = np.expand_dims(Z_init, 0)

    init_state, init_species, init_f = env_option.reset()

    if ODE_env == "Brusselator":
        observation = init_species
    elif ODE_env == "Oregonator":
        observation = init_f
    else:
        observation = init_state
    rc_list = []
    for i in range(max_step):  # while not done:

        if ODE_env == "Oregonator":
            observation[1] = 0.0
            observation[2] = 0.0

        if ODE_env == "Oregonator" and n_episode <= 10:
            u = np.array([0.0, 0.0, 0.0])
        else:
            u = uphill_policy(observation, critic)
        #u = np.array([0,0,0])

        obs, reward, _, _, Z = env_option.step(u)  ##add gaussian noise

        for j in range(len(Z)):
            mu, sigma = 0, 0.001
            # mean and standard deviation
            s = np.random.normal(mu, sigma)
            Z[j] = Z[j] + s

        Z_history = np.concatenate((Z_history, Z), 0)
        states.append(observation)
        observation = obs
        rewards.append(reward)

        #print("observation", observation, "u", u)
        if calc_rate:  # updating every 10 time steps
            theta = rc_model.compute_theta(Z, env.species)
            theta_arr.append(theta)
            Z_arr.append(Z)

            # rc_list.append(estimated_rates)

    # print(rc_list)
    if calc_rate:
        '''
        result = find_rate_constants(Z, Z_arr, theta_arr, rc_model, env_option.u)
        estimated_rates = result.x.tolist()
        print(estimated_rates)
        '''
        result = find_rate_constants(Z_arr, theta_arr, rc_model, env_option.u)
        estimated_rates = result.x.tolist()
        print("estimated_rates", estimated_rates)
        return rewards, states, observation, Z_history, Z, estimated_rates
    else:
        return rewards, states, observation, Z_history, Z, None


def run(env, env_model, ODE_env):
    critic = Critic()

    lr = 0.01

    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)

    print("obs space", env.observation_space)
    print("action_spaction_space", env.action_space.n)

    # train
    max_episode = 2000
    n_episode = 0
    max_step = 200
    N = 10
    scores = []
    prob_list = []
    states_list = []
    returns_list = []

    if ODE_env == "LV":
        rc_model = RateConstantModel(rates=[0, 0, 0], ODE_env=ODE_env)
    elif ODE_env == "Brusselator":
        rc_model = RateConstantModel(num_reactions=4, rates=[0, 0, 0, 0], ODE_env=ODE_env)

    elif ODE_env == "Generalized":
        rc_model = RateConstantModel(num_reactions=6, rates=[0, 0, 0, 0, 0, 0], ODE_env=ODE_env)  # LV

    elif ODE_env == "Oregonator":
        rc_model = RateConstantModel(num_reactions=5, rates=[0, 2.4 * 1e6, 33.6, 2.4 * 1e3, 0],
                                     ODE_env=ODE_env)  # LV

    while n_episode < max_episode:
        theta_arr = []
        Z_arr = []
        print('starting training episode %d' % n_episode)

        if ODE_env == "LV":
            env.rate_constants = [0.1, 0.05, 0.05]  # LV
        elif ODE_env == "Brusselator":
            env.rate_constants = [1, 1, 1, 1]  # Brusselator

        elif ODE_env == "Generalized":
            env.rate_constants = [0.1, 0.05, 0.05, 0, 0, 0]  # LV

        elif ODE_env == "Oregonator":
            env.rate_constants = [1.28, 2.4 * 1e6, 33.6, 2.4 * 1e3, 1]
            # env.rate_constants = [1000, 1000, 1000, 1000, 1000]
            #env.rate_constants =  [1, 2 * 1e9, 5000, 5 * 1e7, 1]

        if n_episode % N == 0:
            if ODE_env == "Oregonator" and n_episode > 10:
                env_option = env
                calc_rate = False

            else:
                env_option = env
                calc_rate = True
                #calc_rate = False
        else:
            env_option = env_model
            calc_rate = False

        rewards, states, observation, Z_history, Z, estimated_rates = run_one_episode(env_option, max_step, critic,
                                                                                      Z_arr, theta_arr,
                                                                                      rc_model, calc_rate, n_episode)
        if ODE_env == "Oregonator" and n_episode == 0:
            estimated = estimated_rates

        if n_episode % N == 0:
            if ODE_env == "Oregonator":
                env_model.rate_constants = estimated
            else:
                env_model.rate_constants = estimated_rates

        print("rate constant", env_option.rate_constants)
        scores.append(sum(rewards))

        n_episode += 1

        R = critic(torch.tensor(observation, dtype=torch.float)).detach().numpy()[0]

        returns = discounted_rewards(rewards, R, gamma=0.4)
        states = states[:-25]
        returns = returns[:-25]

        update_policy(states, returns, critic, critic_optimizer)
        for s in range(len(states)):
            states_list.append(states[s].tolist())
            returns_list.append(returns[s])

    # eval -- let's make this a separate function, analogous to 'run' but without any training or policy updating
    # done = False
    theta_arr = []
    Z_arr = []

    rewards, states, observation, Z_history, Z, estimated_rates = run_one_episode(env, max_step, critic, Z_arr,
                                                                                  theta_arr, rc_model, False, 401)
    print("observation", observation)
    print("Z_history", Z_history)
    # result = find_rate_constants(Z, Z_arr, theta_arr, rc_model, env.u)
    # print("result", result)

    plt.figure(1)
    plt.cla()
    scores = scores[::10]
    print("scores", scores)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.yscale('symlog')
    plt.ylabel('Cumulative reward')
    plt.xlabel('Episode #')
    plt.savefig('./rewards.png')

    if ODE_env != "Oregonator":
        tt = np.linspace(0, max_step, Z_history.shape[0])
        plt.cla()
        plt.plot(tt, Z_history[:, 0], 'b', label='X')
        plt.plot(tt, Z_history[:, 1], 'r', label='Y')

        plt.plot(tt, Z_history[:, 0] / Z_history[:, 1], 'k', label='ratio')
        plt.legend(shadow=True, loc='upper right')
        plt.xlabel('t')
        plt.ylabel('n')
        plt.savefig('./state_trajectory.png')

    if ODE_env == "Brusselator":
        plt.cla()
        A = list(zip(*states))[0]
        B = list(zip(*states))[1]
        plt.plot(np.arange(1, max_step + 1), A, 'b', label='A')
        plt.plot(np.arange(1, max_step + 1), B, 'r', label='B')
        plt.legend(shadow=True, loc='upper right')
        plt.xlabel('t')
        plt.ylabel('n')
        plt.savefig('./species_constants_trajectory.png')
        # plt.show()

        # plt.show()

    if ODE_env == "Oregonator":
        print("hello")
        plt.cla()
        f = list(zip(*states))[0]
        plt.plot(np.arange(1, max_step + 1), f, 'b', label='f')
        plt.legend(shadow=True, loc='upper right')
        plt.xlabel('t')
        plt.ylabel('n')
        plt.savefig('./f_constants_trajectory.png')

    if ODE_env == "Oregonator":
        tt = np.linspace(0, max_step, Z_history.shape[0])
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

        ax1.plot(tt, Z_history[:, 0], 'b', label='X')
        ax1.set_yscale('log')
        ax1.legend(shadow=True, loc='upper right')
        ax2.plot(tt, Z_history[:, 1], 'r', label='Y')
        ax2.legend(shadow=True, loc='upper right')
        ax2.set_yscale('log')
        ax3.plot(tt, Z_history[:, 2], 'k', label='Z')
        ax3.legend(shadow=True, loc='upper right')
        ax3.set_yscale('log')
        #plt.show()

        plt.xlabel('t')
        plt.ylabel('n')
        plt.savefig('./oregonator.png')
        # plt.show()


    if ODE_env != "Oregonator":

        x = np.arange(0.5, 1.5, 0.01)
        y = np.arange(1, 3, 0.02)
        X, Y = np.meshgrid(x, y)

        V = []
        for i in x:
            V_vec = []
            for j in y:
                obs = torch.tensor([i, j], dtype=torch.float)
                val = critic(obs)
                V_vec.append(val.detach().numpy())
            V.append([V_vec])

        V = np.array(V).reshape(100, 100)
        fig, ax = plt.subplots()
        CS = ax.contour(X, Y, V)
        ax.clabel(CS, inline=True, fontsize=10)
        ###
        ax.plot(Z_history[:, 0], (Z_history[:, 1]), label = 'Phase space')
        ax.legend(shadow=True, loc='upper right')
        ###
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.title("Value function")
        plt.savefig('./Value_arr_contour_plot.png')

        X = []
        Y = []
        for s in states_list:
            X.append(s[0])
            Y.append(s[1])

        X = np.array(X)
        Y = np.array(Y)
        returns_list = np.array(returns_list)

        fig, ax = plt.subplots()

        z = ax.tricontour(X, Y, returns_list, 20)
        fig.colorbar(z)
        ax.tricontour(X, Y, returns_list, 20)

        ax.plot(X, Y)
        ax.set_xlabel("Prey")
        ax.set_ylabel("Predators")
        print("hello")
        plt.savefig('./returns.png')

        fig, ax = plt.subplots()
        ax.plot(Z_history[:, 0], (Z_history[:, 1]))
        ax.set_xlabel("Prey")
        ax.set_ylabel("Predators")
        plt.savefig('./Z_history_contour_plot.png')


    else:
        fig = plt.figure()
        ax = Axes3D(fig)

        ax.plot(Z_history[:, 0], Z_history[:, 1], Z_history[:, 2])  # For line plot

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.savefig('./Z_history_contour_plot.png')

    # plt.show()
    exit()
    env.close()


N = 2  # number of species
tau = 1
dt = 0.01  # 1e-2

#ODE_env = "LV"
ODE_env = "Brusselator"
# ODE_env = "Generalized"
#ODE_env = "Oregonator"

if ODE_env == "LV":
    env = LotkaVolterraEnv(N, tau, dt)
    env_model = LotkaVolterraEnvModel(N, tau, dt)
elif ODE_env == "Brusselator":
    env = BrusselatorEnv(N, tau, dt)
    env_model = BrusselatorEnvModel(N, tau, dt)
elif ODE_env == "Generalized":
    env = GeneralizedEnv(N, tau, dt)
    env_model = GeneralizedEnvModel(N, tau, dt)
elif ODE_env == "Oregonator":
    N = 3  # number of species
    env = OregonatorEnv(N, tau, dt)
    env_model = OregonatorEnvModel(N, tau, dt)
# run(env, algorithm = "reinforce")
run(env, env_model, ODE_env)
# run(env, algorithm = "optimal policy")