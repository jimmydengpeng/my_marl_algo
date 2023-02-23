import gym
import highway_env
from deep_rl import *
from colorlog import logger

# logger.debug("env.config:", env.config)
# env.config["action"].update(
#     {
#         "lateral": False,
#         "longitudinal": True,
#         "type": "DiscreteMetaAction"
#         # "type": "ContinuousAction"
#     }
# )

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

def gym_like_test():
    env = gym.make("intersection-v0")
    env.config.update({
        "simulation_frequency": 5
    })
    # logger.debug("env.action_space", env.action_space)
    logger.debug("env.config:", env.config)

    # logger.debug("env.action_space", env.action_space)

    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        logger.debug("action:", action)
        obs, reward, done, info = env.step(action)
        logger.debug("obs:", obs)
        logger.debug("obs:", type(obs))
        logger.debug("obs:", obs.shape)
        env.render()


if __name__ == "__main__":
    mkdir('log')
    # set_one_thread()
    random_seed()
    game = "intersection-v0"

    dqn_feature(game=game, n_step=1, replay_cls=UniformReplay, async_replay=True, noisy_linear=True)