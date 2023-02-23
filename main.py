from collections import namedtuple
import torch
import gym
from PPOAgent import PPO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Memory = namedtuple('Memory', ['states', 'actions', 'rewards', 'next_states', 'masks'])

def main():
    env = gym.make("highway-v0")
    # env = gym.make('CartPole-v1')
    # env.seed(0)
    torch.manual_seed(0)

    print(type(env.observation_space.shape))
    exit()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    hidden_size = 64
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99
    K_epochs = 4
    eps_clip = 0.2
    T_horizon = 2048
    N_iterations = 1000
    batch_size = 64

    ppo = PPO(state_dim, action_dim, hidden_size, lr, betas, K_epochs, eps_clip)
    memory = []
    for i in range(1, N_iterations+1):
        state = env.reset()
        ep_reward = 0
        for t in range(T_horizon):
            action = ppo.select_action(state)
            next_state, reward, done, _ = env.step(action)
            memory.append(Memory(state, action, reward, next_state, 1-done))
            state = next_state
            ep_reward += reward

            if len(memory) == batch_size:
                ppo.update(memory, gamma)
                memory = []

            if done:
                break

        print(f"Episode {i}, reward: {ep_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()