

# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from ppo_class import *
class Environment:
    def __init__(self, num_classes=12):
        self.num_classes = num_classes
        self.state = np.zeros(self.num_classes)
        self.state[0] = 1.0

    def reset(self):
        self.state = np.zeros(self.num_classes)
        self.state[0] = 1.0
        return self.state

    def step(self, action):
        done = False
        reward = 0

        if action == np.argmax(self.state):
            reward = 10

        self.state = np.roll(self.state, 1)
        next_state = self.state.copy()
        if next_state[0] == 1:
            done = True

        return next_state, reward, done


env = Environment(observation_dimensions)


def model(env):
    # Initialize the observation, episode return and episode length
    print(env.reset())
    observation, episode_return, episode_length = env.reset(), 0, 0
    print(observation)
    epochs = 50
    # Iterate over the number of epochs
    for epoch in range(epochs):
        # Initialize the sum of the returns, lengths and number of episodes for each epoch
        sum_return = 0
        sum_length = 0
        num_episodes = 0
        steps_per_epoch = 1000
        # Iterate over the steps of each epoch
        for t in range(steps_per_epoch):

            # Get the logits, action, and take one step in the environment
            # print(observation)
            observation = observation.reshape(1, -1)
            logits, action = sample_action(observation)
            # print( env.step(action[0].numpy()))
            observation_new, reward, done = env.step(action[0].numpy())
            # print('new ob',observation_new)
            episode_return += reward
            episode_length += 1

            # Get the value and log-probability of the action
            value_t = critic(observation)
            logprobability_t = logprobabilities(logits, action)

            # Store obs, act, rew, v_t, logp_pi_t
            buffer.store(observation, action, reward, value_t, logprobability_t)

            # Update the observation
            observation = observation_new

            # Finish trajectory if reached to a terminal state
            terminal = done
            if terminal or (t == steps_per_epoch - 1):
                last_value = 0 if done else critic(observation.reshape(1, -1))
                buffer.finish_trajectory(last_value)
                sum_return += episode_return
                sum_length += episode_length
                num_episodes += 1
                observation, episode_return, episode_length = env.reset(), 0, 0

        # Get values from the buffer
        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
        ) = buffer.get()

        # Update the policy and implement early stopping using KL divergence
        for _ in range(train_policy_iterations):
            kl = train_policy(
                observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
            )
            if kl > 1.5 * target_kl:
                # Early Stopping
                break

        # Update the value function
        for _ in range(train_value_iterations):
            train_value_function(observation_buffer, return_buffer)

        # Print mean return and length for each epoch
        print(
            f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
        )
    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
    observation, episode_return, episode_length = env.reset(), 0, 0
    done = False
    while not done:
        observation = observation.reshape(1, -1)
        action = get_optimal_action(observation)
        print('at', observation, 'action', action)
        observation_new, _, done = env.step(action)
        observation = observation_new


model(env)