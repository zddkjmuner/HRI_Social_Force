import argparse
import numpy as np
import tensorflow as tf
import time
import pickle

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

""" import mpctools as mpc
import mpctools.plots as mpcplots
import control """

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="zdd_env", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=1000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=128, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=200, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

""" def get_mpc(env):
    agent_r = env.world.agents[0].size
    landmark_r = env.world.landmarks[0].size
    m = 2
    umax = [env.agents[0].u_range, env.agents[0].phi_range]
    xmax = 5


    Nx = 4
    Nu = 2
    Nt = 50
    Nsim = 100

    umax = 3
    xmax = 5
    cushion = 3

    Acont = np.array([
        [0,0,1,0],
        [0,0,0,1],
        [0,0,0,0],
        [0,0,0,0],
    ])
    Bcont = np.array([
        [0,0],
        [0,0],
        [1,0],
        [0,1],
    ])

    Delta = 0.02

    f = mpc.getCasadiFunc(lambda x,u: mpc.mtimes(Acont,x) + mpc.mtimes(Bcont,u),
                      [Nx,Nu],["x","u"], "f")

    model = mpc.DiscreteSimulator(lambda x,u: mpc.mtimes(Acont,x) + mpc.mtimes(Bcont,u), Delta, [Nx,Nu], ["x","u"])

    # stagecost
    def lfunc(x, u):
        return 1000 * mpc.mtimes(x[2:].T, x[2:]) + mpc.mtimes(u.T, u)
    l = mpc.getCasadiFunc(lfunc, [Nx, Nu], ["x","u"], "l")

    # terminal cost
    def termcost(x):
        return 1000 * mpc.mtimes(x[2:].T, x[2:])
    Pf = mpc.getCasadiFunc(termcost, [Nx], ['x'], funcname='Pf')

    rmin = 0.2
    def termcon(x):
        return x[2] ** 2 + x[3] ** 2
    ef = mpc.getCasadiFunc(termcon, [Nx], ['x'], funcname='ef') """

    # center = np.random.randn(m,2)
    # center = np.array([[1.5,1.5], [2.5, 2.5]])
    # center = [(center[i,0], center[i,1]) for i in range(center.shape[0])]
    # holes = [(p, r) for p in center]

    env.reset()
    x0 = np.zeros(4)
    # env.agents[0].state.p_vel = np.array([0.5, 0.5])
    x0[:2], x0[2:] = env.agents[0].state.p_pos, env.agents[0].state.p_vel

    centers = [(landmark.state.p_pos[0], landmark.state.p_pos[1]) for landmark in env.world.landmarks[1:]]

    slack = 0.2
    def nlcon(x):
        # [x1, x2] = x[0:2] # Doesn't work in Casadi 3.0
        x1 = x[0]
        x2 = x[1]
        resid = [agent_r**2 + landmark_r**2 + slack - (x1 - p1)**2 - (x2 - p2)**2 for (p1,p2) in centers]
        return np.array(resid)
    e = mpc.getCasadiFunc(nlcon, [Nx], ["x"], "e")

    """ movingHorizon = True
    terminalConstraint = True
    terminalWeight = False

    # Build Optimizer
    if not movingHorizon:
        Nt = Nsim
        Nsim = 1

    lb = {
        "u": -umax*np.ones((Nt,Nu)),
        "x": np.tile([-cushion + slack,-cushion + slack,-np.inf,-np.inf], (Nt+1,1)),
    }
    ub = {
        "u" : umax*np.ones((Nt,Nu)),
        "x" : np.tile([xmax+cushion - slack,xmax+cushion - slack,np.inf,np.inf], (Nt+1,1)),
    }

    guess = {"x" : np.tile(x0, (Nt+1,1))}

    if movingHorizon:
        verb = 0
    else:
        verb = 5

    Ne = len(centers)
    N = {"t":Nt, "x":Nx, "u":Nu, "e":Ne, 'c':2}

    funcargs = {"f" : ["x","u"], "e" : ["x"], "l" : ["x","u"], "ef" : ["x"]}

    nominal_mpc = mpc.nmpc(f, l, N, x0, lb, ub, funcargs=funcargs, e=e, Pf=Pf,
                      ef=ef, verbosity=(0 if movingHorizon else 5),
                      casaditype="SX", Delta=Delta)

    nominal_mpc.initialize(solveroptions=dict(max_iter=10000))

    [X, L, G] = control.care(A=Acont, B=Bcont, Q=np.eye(4), R=np.eye(2))

    x = np.zeros((Nsim+1,Nx))
    actual = np.zeros((Nsim+1,Nx))
    x[0,:] = x0
    actual[0,:] = x0
    u = np.zeros((Nsim,Nu))
    feedback = np.zeros((Nsim,Nu))

    return nominal_mpc, G, x """

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers
                                                        

def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        nominal_mpc, G, x = get_mpc(env)

        print('Starting iterations...')
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            done = False
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
