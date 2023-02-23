import gym
import highway_env
from deep_rl import *

def dqn_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('n_step', 1)
    kwargs.setdefault('replay_cls', UniformReplay)
    kwargs.setdefault('async_replay', True)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: VanillaNet(config.action_dim, FCBody(config.state_dim))
    # config.network_fn = lambda: DuelingNet(config.action_dim, FCBody(config.state_dim))
    config.history_length = 1
    config.batch_size = 10
    config.discount = 0.99
    config.max_steps = 1e5

    replay_kwargs = dict(
        memory_size=int(1e4),
        batch_size=config.batch_size,
        n_step=config.n_step,
        discount=config.discount,
        history_length=config.history_length)

    config.replay_fn = lambda: ReplayWrapper(config.replay_cls, replay_kwargs, config.async_replay)
    config.replay_eps = 0.01
    config.replay_alpha = 0.5
    config.replay_beta = LinearSchedule(0.4, 1.0, config.max_steps)

    config.random_action_prob = LinearSchedule(1.0, 0.1, 1e4)
    config.target_network_update_freq = 200
    config.exploration_steps = 1000
    # config.double_q = True
    config.double_q = False
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.eval_interval = int(5e1)
    config.async_actor = False
    config.num_workers = 1
    
    # print((config))
    # exit()
    run_steps(DQNAgent(config))

def ppo_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)
    env_config = {
        "simulation_frequency": 2,
        "action": {
                # "type": "DiscreteMetaAction",
                "type": "ContinuousAction", # TODO
                # "lateral": False
                # "longitudinal": False
            },
    }
    config.task_fn = lambda: Task(
        config.game,
        env_config=env_config,
        num_envs=8,
        single_process=False
    )
    config.eval_env = config.task_fn()

    config.network_fn = lambda: GaussianActorCriticNet(
        config.state_dim, config.action_dim, actor_body=FCBody(config.state_dim, gate=torch.tanh),
        critic_body=FCBody(config.state_dim, gate=torch.tanh))
    config.actor_opt_fn = lambda params: torch.optim.Adam(params, 3e-4)
    config.critic_opt_fn = lambda params: torch.optim.Adam(params, 1e-3)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.rollout_length = 512
    # config.rollout_length = 2048
    config.optimization_epochs = 10
    config.mini_batch_size = 64
    config.ppo_ratio_clip = 0.2
    # config.log_interval = 2048
    config.log_interval = 256
    # config.max_steps = 3e6
    config.max_steps = 3e5
    config.target_kl = 0.01
    config.state_normalizer = MeanStdNormalizer()
    config.num_workers = 8
    run_steps(PPOAgent(config))


if __name__ == "__main__":
    mkdir('log')
    # set_one_thread()
    random_seed()
    # select_device(-1)

    # env = gym.make("highway-v0")
    # print(env.action_space)
    # exit()


    game = "highway-v0"
    # game = "intersection-v0"
    # game = "parking-v0"


    # game = 'CartPole-v0'
    # print(env.default_config()['action']['type'])
    # env.default_config()['action']['type'] = "ContinuousAction"
    # print(env.default_config()['action']['type'])
    # env.config['action']['type'] = 'ContinuousAction'
    # print(env.config['action']['type'])
    # print(env.default_config())

    # dqn_feature(game=game, n_step=1, replay_cls=UniformReplay, async_replay=True, noisy_linear=True)
    ppo_continuous(game=game)
    # env.reset()

    # obs = env.reset()
    # done = False
    # while not done:
    #     action = env.action_space.sample()
    #     print(action)
    #     obs, reward, done, info = env.step(action)
    #     env.render()