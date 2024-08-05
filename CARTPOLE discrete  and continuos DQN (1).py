#!/usr/bin/env python
# coding: utf-8

# # Part 1
# 

# ## CartPole-Discrete Action Space and Continuous Observation Space

# In[2]:


import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import  evaluate_policy
import os


# ## DQN Policy

# In[5]:


models_dir = "models/DQN_Cartpole"
logdir = "logs1"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)
    


# In[6]:


env = gym.make("CartPole-v1")
model =DQN("MlpPolicy",env,verbose=1,tensorboard_log=logdir)
model.learn(total_timesteps=500000)


# In[5]:


evaluate_policy(model,env,n_eval_episodes=10,render=True)


# In[6]:


model.save("CartPole DQN Model")


# In[7]:


env = gym.make("CartPole-v1",render_mode = 'human')

model.load("Cartpole DQN Model")
episodes = 20
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



# In[10]:


print(total_re)
plt.figure(figsize=(15, 10))
plt.plot( total_re)
plt.xlabel('Episode', fontsize=28)
plt.ylabel('Reward Value', fontsize=28)
plt.title('Rewards Per Episode DQN CartPole Test', fontsize=36)
plt.ylim(ymin=0, ymax=200)
plt.xlim(xmin=0, xmax= 20)
plt.grid()
plt.show()


# In[1]:


import psutil

# Get CPU usage as a percentage
cpu_usage = psutil.cpu_percent()

print(f"CPU Usage: {cpu_usage}%")


# In[6]:


import matplotlib.pyplot as plt
import gym
from stable_baselines3 import DQN  # Replace with the actual import for your DQN model


env = gym.make("CartPole-v1", render_mode='human')
model = DQN.load("Cartpole DQN Model")
episodes = 30
total_rewards = []
episode_numbers = list(range(1, episodes + 1))

for ep in range(episodes):
    obs, _ = env.reset()
    episode_reward = 0
    done = False

    while not done:
        
        action, _ = model.predict(obs)



        obs, reward, done, _ , _ = env.step(action)
        episode_reward += reward

    total_rewards.append(episode_reward)
    print(f'Episode {ep + 1}: Total Reward: {episode_reward}')

env.close()

#the convergence history
plt.plot(episode_numbers, total_rewards, marker='o', linestyle='-')
plt.xlabel('Episode Number')
plt.ylabel('Total Reward')
plt.title('Convergence History for CartPole DQN Model')
plt.show()


# In[ ]:




