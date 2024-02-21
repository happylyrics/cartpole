import gymnasium as gym

goal = 0
last_error = 0
kp = 1
kd = 5
env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    cart_position = observation[2]
    error = goal - cart_position
    dif = error - last_error
    controlOutput = kp * error + kd * dif
    last_error = error
    if controlOutput < 0:
        action = 1
    else:
        action = 0
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
env.close()