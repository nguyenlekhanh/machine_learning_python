import numpy as np
import gym

# Tạo môi trường FrozenLake-v1
env = gym.make('FrozenLake-v1', is_slippery=False)  # Set is_slippery=False để làm môi trường ít phức tạp hơn

# Khởi tạo Q-table
state_space_size = env.observation_space.n
action_space_size = env.action_space.n
Q = np.zeros((state_space_size, action_space_size))

# Tham số học của thuật toán Q-Learning
learning_rate = 0.8
discount_factor = 0.95
num_episodes = 1000

# Hàm chọn hành động dựa trên chiến lược epsilon-greedy
def choose_action(state):
    if np.random.rand() < 0.5:  # Exploration
        return env.action_space.sample()
    else:  # Exploitation
        return np.argmax(Q[state, :])

# Huấn luyện với thuật toán Q-Learning
for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        action = choose_action(state)
        next_state, reward, done = env.step(action)

        # Cập nhật Q-value bằng phương trình Bellman
        Q[state, action] = Q[state, action] + learning_rate * (reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action])

        state = next_state

# Kiểm thử mô hình đã học
num_test_episodes = 10
total_rewards = 0

for _ in range(num_test_episodes):
    state = env.reset()
    done = False

    while not done:
        action = np.argmax(Q[state, :])
        next_state, reward, done = env.step(action)
        total_rewards += reward
        state = next_state

average_reward = total_rewards / num_test_episodes

print(f'Average reward over {num_test_episodes} test episodes: {average_reward}')
