import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.actor(x), self.critic(x)

class PPO:
    def __init__(self, state_dim, action_dim, hidden_size, lr, betas, K_epochs, eps_clip):
        self.policy = ActorCritic(state_dim, action_dim, hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip

    def select_action(self, state):
        state = torch.FloatTensor(state)
        logits, value = self.policy(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item()

    def update(self, memory, gamma):
        states = torch.FloatTensor(memory.states)
        actions = torch.LongTensor(memory.actions)
        rewards = torch.FloatTensor(memory.rewards)
        next_states = torch.FloatTensor(memory.next_states)
        masks = torch.FloatTensor(memory.masks)

        old_logits, old_values = self.policy(states)
        old_dist = Categorical(logits=old_logits)
        old_log_probs = old_dist.log_prob(actions)

        returns = torch.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return * masks[t]
            returns[t] = running_return

        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        for _ in range(self.K_epochs):
            logits, values = self.policy(states)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratios = torch.exp(log_probs - old_log_probs)
            advantages = returns - old_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(returns, values)

            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
