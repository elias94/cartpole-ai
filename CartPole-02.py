import gym

EPISODES = 1000
avg_time = 0
max_time = -1
env = gym.make('CartPole-v0')

def action_policy(state):
    return 0 if state[2] < 0 else 1

for i_episode in range(EPISODES):
    state = env.reset()
    for t in range(1000):
        # env.render()
        action = action_policy(state)
        state, reward, done, info = env.step(action)

        if done:
            avg_time = avg_time + t
            if t >max_time:
                max_time = t
                print('Maximum time reached:', max_time)
            break

print('AVG time network survives : ', avg_time/EPISODES)
