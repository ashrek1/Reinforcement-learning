import numpy as np
import math
import gym
import matplotlib.pyplot as plt

def test(EPISODES, show, table_size, learning_rate, discount, epsilon, start_epsililon=1):
    '''
    
    EPISODES:
    the number of times thhe simulation will run
    
    show:
    when the simulation will render and the checkpoint for metrics
    
    table_size:
    the number of buckets in the q-learning table 
    
    learning_rate:
    the learning rate
    
    discount
    
    
    epsilon 
    start_epsililon 
    '''
    env = gym.make('CartPole-v0')
    
    high = env.observation_space.high
    high[[1,3]] = 5
    low = env.observation_space.low
    low[[1,3]] = -5
    
    def get_discrete_state(state):
        return tuple(((state - low) / os_win).astype(np.int))
    
    os_size = [table_size] * len(high)
    os_win = (high - low) / os_size

    end_epsilion = EPISODES // 2

    epsilon_decay_value = epsilon/(end_epsilion - start_epsililon)

    q_table = np.random.uniform(low=-1, high=0, size=(os_size + [env.action_space.n]))
    
    ep_rewards = []
    agg_ep_rewards = {'ep':[],'avg':[], 'min':[], 'max':[]}

    for episode in range(EPISODES):
        if episode % show == 0:
            #print(episode)
            render = True
        else:
            render= False

        discrete_state = get_discrete_state(env.reset())
        episode_reward = 0
        done = False

        while not done:
            if render == True:
                env.render()
                
            if np.random.random() > epsilon:
                action = np.argmax(q_table[discrete_state])
            else:
                action = np.random.randint(0, env.action_space.n)
            
            action = np.argmax(q_table[discrete_state])
            new_state, reward, done, _ = env.step(action)

            episode_reward += reward
            new_discrete = get_discrete_state(new_state)

            max_future_q = np.max(q_table[new_discrete])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1-learning_rate) * current_q + learning_rate * (reward+discount*max_future_q)
            q_table[discrete_state+(action,)] = new_q

            discrete_state = new_discrete

        if end_epsilion >= episode >= start_epsililon:
            epsilon -= epsilon_decay_value
            
        ep_rewards.append(episode_reward)
        
        if not episode % show:
            average_reward = sum(ep_rewards[-show:])/len(ep_rewards[-show:])
            agg_ep_rewards['ep'].append(episode)
            agg_ep_rewards['avg'].append(average_reward)
            agg_ep_rewards['min'].append(min(ep_rewards[-show:]))
            agg_ep_rewards['max'].append(max(ep_rewards[-show:]))
            
            #print('episode: {}, avg: {}, min: {}, max: {}'.format(episode, average_reward, min(ep_rewards[-show:]), max(ep_rewards[-show:])))

    env.close()
    
    plt.plot(agg_ep_rewards['ep'], agg_ep_rewards['avg'], label='avg', color='black')
    plt.plot(agg_ep_rewards['ep'], agg_ep_rewards['min'], label='min', color='green')
    plt.plot(agg_ep_rewards['ep'], agg_ep_rewards['max'], label='max', color='red')
    plt.legend()
    return agg_ep_rewards['avg']
