#!/usr/bin/env python
# coding: utf-8

# # Part 1

# ## FrozenLake - Discrete Action and Observation Space

# In[1]:


import gymnasium as gym 
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import  evaluate_policy
import os


# ## DQN Policy

# In[5]:


models_dir = "models/DQN_FrozenLake"
logdir = "logs"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)
    


# In[6]:


env = gym.make('FrozenLake-v1', desc=None, map_name="4x4")
model =DQN("MlpPolicy",env,verbose=1,tensorboard_log=logdir)
model.learn(total_timesteps=500000)


# In[4]:


evaluate_policy(model,env,n_eval_episodes=10,render=True)


# In[5]:


model.save("Frozenlake DQN Model")


# In[6]:


import gymnasium as gym
import os



# Create the FrozenLake environment
env = gym.make("FrozenLake-v1",render_mode='human')

# Load the DQN model
model = DQN.load("Frozenlake DQN Model")

episodes = 30
obs,_ = env.reset()
re = 0
total_re = []

for ep in range(episodes):
    re = 0
    obs, _ = env.reset()
    done = False
    while not done:
        # Predict a continuous action
        action, _ = model.predict(obs)
        
        # Round the continuous action to the nearest integer
        action = int((action))

        obs, reward, done, _ , _= env.step(action)
        re += reward
    total_re.append(re)
    print(f'episode {ep} and reward is {re}')
env.close()


# In[8]:


print(total_re)
plt.figure(figsize=(12, 10))
plt.plot( total_re)
plt.xlabel('Episode', fontsize=18)
plt.ylabel('Reward', fontsize=18)
plt.title('Rewards Per Episode DQN FrozenLake Test', fontsize=36)
plt.ylim(ymin=-1, ymax=3)
plt.xlim(xmin=0, xmax= 30)
plt.grid()
plt.show()


# In[9]:


import psutil

# Get CPU usage as a percentage
cpu_usage = psutil.cpu_percent()

print(f"CPU Usage: {cpu_usage}%")


# In[12]:


import matplotlib.pyplot as plt
import gym
from stable_baselines3 import DQN  # Replace with the actual import for your DQN model


env = gym.make("FrozenLake-v1", render_mode='human')


model = DQN.load("Frozenlake DQN Model")
episodes = 30
total_rewards = []
episode_numbers = list(range(1, episodes + 1))

for ep in range(episodes):
    obs, _ = env.reset()
    episode_reward = 0
    done = False

    while not done:
        # Predict a continuous action
        action, _ = model.predict(obs)

        # Round the continuous action to the nearest integer
        action = int(action)

        obs, reward, done, _ , _ = env.step(action)
        episode_reward += reward

    total_rewards.append(episode_reward)
    print(f'Episode {ep + 1}: Total Reward: {episode_reward}')

env.close()

#the convergence history
plt.plot(episode_numbers, total_rewards, marker='o', linestyle='-')
plt.xlabel('Episode Number')
plt.ylabel('Total Reward')
plt.title('Convergence History for FrozenLake-v1')
plt.show()


# In[ ]:




