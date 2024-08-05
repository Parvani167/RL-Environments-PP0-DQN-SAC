#!/usr/bin/env python
# coding: utf-8

# # Part 2

# ## LunarLander - Continuous Action And Observation Space

# In[4]:


import gym


# In[5]:


pip install box2d box2d-kengz


# # Continuous Environment - SAC Policy

# In[2]:


import gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import  evaluate_policy
import os


# In[3]:


models_dir = "models/SAC_LunarLander"
logdir = "logs2"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)
    


# In[4]:


# Create the LunarLander environment
env = gym.make("LunarLanderContinuous-v2")#, render_mode="human")


# In[5]:


# env = gym.make("LunarLander-v2", render_mode="human")
# env = DummyVecEnv([lambda: env])
model =SAC("MlpPolicy",env,verbose=1,tensorboard_log=logdir)
model.learn(total_timesteps=500000)


# In[12]:


evaluate_policy(model,env,n_eval_episodes=10,render=True)


# In[13]:


model.save("Lunar SAC Model")


# In[14]:


env = gym.make("LunarLanderContinuous-v2",render_mode = 'human')

model.load("Lunar SAC Model")
episodes = 30
obs,_ = env.reset()
re = 0
total_re = []

for ep in range(episodes):
    re = 0
    obs, _ = env.reset()
    done = False
    while not done:
        
        action,_ = model.predict(obs)
        obs,reward, done, _, _ = env.step(action)
        re+= reward
    total_re.append(re)
    print(f'episode {ep} and reward is {re}')
env.close()






# In[19]:


plt.figure(figsize=(20, 10))
plt.plot( total_re)
plt.xlabel('Episode', fontsize=13)
plt.ylabel('Reward', fontsize=28)
plt.title('Rewards Per Episode SAC LunarLander Test', fontsize=36)
plt.ylim(ymin=0, ymax=400)
plt.xlim(xmin=0, xmax= 32)
plt.grid()
plt.show()


# In[20]:


import psutil

# Get CPU usage as a percentage
cpu_usage = psutil.cpu_percent()

print(f"CPU Usage: {cpu_usage}%")


# In[24]:


import matplotlib.pyplot as plt
import gym


env = gym.make("LunarLanderContinuous-v2", render_mode='human')


episodes = 30


total_rewards = []
episode_numbers = list(range(1, episodes + 1))

for ep in range(episodes):
    obs, _ = env.reset()
    episode_reward = 0
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        episode_reward += reward

    total_rewards.append(episode_reward)
    print(f'Episode {ep + 1}: Total Reward: {episode_reward}')

env.close()

#the convergence history
plt.plot(episode_numbers, total_rewards, marker='o', linestyle='-')
plt.xlabel('Episode Number')
plt.ylabel('Total Reward')
plt.title('Convergence History for LunarLanderContinuous-v2')
plt.show()


# In[ ]:




