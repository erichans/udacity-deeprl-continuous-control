from agent import DDPGAgent

from collections import deque

from unityagents import UnityEnvironment

import torch
import numpy as np

import matplotlib.pyplot as plt

import datetime

def start_env():
    env = UnityEnvironment(file_name='Reacher_Windows_20_agent_x86_64/Reacher.exe')

    # get the default brain
    brain_name = get_brain_name(env)
    brain = get_brain(env)

    env_info = reset_env_info(env)

    print('Number of agents:', get_total_agents(env_info))

    action_size = get_action_size(env)
    print('Number of actions:', action_size)

    state = env_info.vector_observations[0]
    print('States look like:', state)
    print('States have length:', get_state_size(env_info))
    
    return env

def get_brain_name(env):
    return env.brain_names[0]
    
def get_brain(env):
    return env.brains[get_brain_name(env)]
    
def get_state_size(env_info):
    return len(env_info.vector_observations[0])
    
def get_action_size(env):
    return get_brain(env).vector_action_space_size
    
def reset_env_info(env):
    return env.reset(train_mode=True)[get_brain_name(env)]
    
def env_step(env, action):
    return env.step(action)[get_brain_name(env)]

def get_total_agents(env_info):
    return len(env_info.agents)

def ddpg_run(episodes=1000, seed=42):
    env = start_env()
    env_info = reset_env_info(env)
    
    state_size = get_state_size(env_info)
    action_size = get_action_size(env)
    
    print('Seed used:', seed)
    total_agents = get_total_agents(env_info)
    agent = DDPGAgent(total_agents, state_size, action_size, seed)
    
    scores = []
    scores_window = deque(maxlen=100)
    for episode in range(1, episodes+1):
        init_time = datetime.datetime.now()
        
        env_info = reset_env_info(env)
        score = np.zeros(total_agents)
        dones = np.zeros(total_agents)
        agent.reset()
        critic_losses = []
        actor_losses = []
        while not np.any(dones):
            states = env_info.vector_observations
            actions = agent.act(states, add_noise=True)
            env_info = env_step(env, actions)
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            critic_loss, actor_loss = agent.step(states, actions, rewards, next_states, dones)
            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)
            #print('\rActor Loss: {:.6f} - Critic Loss: {:.6f}'.format(actor_loss, critic_loss), end='')
                
            score += rewards

        scores_window.append(np.mean(score))
        scores.append(np.mean(score))
        print('Ep. {}/{} - Avg Global Score: {:.2f} - Avg Ep. Score: {:.2f} - Min Ep. Score: {:.2f} - Max Ep. Score: {:.2f} - Actor loss: {:.6f}, Critic loss: {:.6f} - time: {}'.format(episode, episodes,
            np.mean(scores_window), np.mean(score), np.min(score), np.max(score), np.mean(actor_losses), np.mean(critic_losses), datetime.datetime.now() - init_time, end=' '))
            
        if np.mean(scores_window) >= 30.0 and episode >= 100:
            print('\nEnvironment solved (mean of 30.0 for 100 episodes) in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
            
            torch.save(agent.actor_local.state_dict(), 'actor_local_checkpoint.pth')
            torch.save(agent.actor_target.state_dict(), 'actor_target_checkpoint.pth')
            
            torch.save(agent.critic_local.state_dict(), 'critic_local_checkpoint.pth')
            torch.save(agent.critic_target.state_dict(), 'critic_target_checkpoint.pth')
            break
    
    env.close()
    return scores
    
def save_scores(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.xlabel('Episode #')
    plt.ylabel('Score')
    plt.savefig('score-evolution.png')
    #plt.show()

TOTAL_EPISODES = 200
SEED = 0

if __name__ == '__main__':    
    scores = ddpg_run(TOTAL_EPISODES, SEED)
    save_scores(scores)