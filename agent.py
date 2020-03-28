import torch
import torch.optim as optim
import torch.nn.functional as F

import random
import numpy as np

from buffer import UniformReplayBuffer
from model import Actor, Critic
from noise import OrnsteinUhlenbeckProcess, LinearSchedule

BUFFER_SIZE = int(1e6) # *** 1e6 Paper
BATCH_SIZE = 64 # *** 64 - paper times total_agents

GAMMA = .99

TAU = 1e-3

LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
WEIGHT_DECAY = 0 # *** 1e-2 Paper - L2 weight decay

class DDPGAgent:
    def __init__(self, total_agents, state_size, action_size, seed):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        #self.device = 'cpu'
        
        self.total_agents = total_agents
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        self.actor_local = Actor(self.state_size, self.action_size, seed).to(self.device)
        self.actor_target = Actor(self.state_size, self.action_size, seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        
        self.critic_local = Critic(self.state_size, self.action_size, seed).to(self.device)
        self.critic_target = Critic(self.state_size, self.action_size, seed).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        #self.noise = OrnsteinUhlenbeckNoise(action_size, seed)
        self.noise = OrnsteinUhlenbeckProcess((self.total_agents, action_size), std=LinearSchedule(0.2))
        
        self.replay_buffer = UniformReplayBuffer(BUFFER_SIZE, BATCH_SIZE * self.total_agents, seed, self.device)
        #self.replay_buffer = PrioritizedReplay(BUFFER_SIZE, self.device)
        
        print('Device used: {}'.format(self.device)) 
        
        print('Actor Local DDPG ->', self.actor_local)
        print('Actor Target DDPG ->', self.actor_target)
        
        print('Critic Local DDPG ->', self.critic_local)
        print('Critic Target DDPG ->', self.critic_target)
        
    def reset(self):
        self.noise.reset()
    
    def act(self, states, add_noise=False):
        states = torch.from_numpy(states).float().to(self.device)
        
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        
        return np.clip(actions + self.noise.sample(), -1, 1) if add_noise else actions
    
    def step(self, states, actions, rewards, next_states, dones):
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.replay_buffer.add(state, action, reward, next_state, done)
            
            #for _ in range(self.total_agents): TOO SLOW
        #if len(self.replay_buffer) > BATCH_SIZE:
        return self._learn(self.replay_buffer.sample(), GAMMA)
        
        #return (None,None)
    
    def _learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        
        # ---------- CRITIC UPDATE --------------------
        next_actions = self.actor_target(next_states)
        next_rewards = self.critic_target(next_states, next_actions)
        target_rewards = rewards + gamma * next_rewards * (1 - dones)
        predicted_rewards = self.critic_local(states, actions)
        
        critic_loss = F.mse_loss(predicted_rewards, target_rewards)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        # ---------- ACTOR UPDATE --------------------    
        predicted_actions = self.actor_local(states)
        actor_loss = -self.critic_local(states, predicted_actions).mean()
        #print('\rActor Loss: {:.6f} - Critic Loss: {:.6f}'.format(actor_loss, critic_loss), end='')
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.critic_local, self.critic_target, TAU)
        self._soft_update(self.actor_local, self.actor_target, TAU)
        
        return critic_loss.cpu().data.numpy(), actor_loss.cpu().data.numpy()
    
    def _soft_update(self, local_model, target_model, tau):
        for local_parameter, target_parameter in zip(local_model.parameters(), target_model.parameters()):
            target_parameter.data.copy_((1.0-tau)*target_parameter+(tau*local_parameter))