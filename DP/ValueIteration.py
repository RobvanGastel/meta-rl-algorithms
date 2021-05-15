import argparse
import copy
import gym
import numpy as np

"""Implement on the environment https://gym.openai.com/envs/FrozenLake-v0/
"""


def value_iteration(env, discount_factor=1.0,
                    theta=1e-3):
    def next_actions(s, V):
        actions = np.zeros(env.nA)
        for action in range(env.nA):
            for prob, next_state, reward, _ in env.P[s][action]:
                actions[action] += prob * (reward + discount_factor
                                           * V[next_state])
        return actions

    # Define an equiprobable policy over all states
    policy = np.ones([env.nS, env.nA]) / env.nA
    iteration = 0
    V = np.zeros(env.nS)

    while True:
        delta = 0

        # We can update our policy in the same loop
        for s in range(env.nS):
            v = copy.deepcopy(V[s])
            v_values = next_actions(s, V)
            best_action = np.max(v_values)
            V[s] = best_action

            # Update the policy with the best action
            policy[s] = np.eye(env.nA)[np.argmax(v_values)]

            delta = max(delta, np.abs(v-V[s]))
        if delta < theta:
            break
        # Print out iterations to compare with Policy
        iteration += 1
        print("Policy:", policy)
        print("Value function:", V)
        print("iteration:", iteration)

    return np.array(V), policy


def act_in_environment(env, policy, render=False):
    state = env.reset()
    episode_rew = 0

    for step in range(env._max_episode_steps):
        action = np.argmax(policy[state])
        state, reward, done, _ = env.step(action)

        if render:
            env.render()

        episode_rew += reward

        if done:
            print(
                f"Finished the episode with {episode_rew} \
                    reward in {step} steps")
            break
    return step, episode_rew


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Execute policy iteration on the FrozenLake environment.')
    parser.add_argument("--slippery", action='store_true')
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--render", default=True, type=bool)
    args = parser.parse_args()

    # Execute policy iteration
    env = gym.make('FrozenLake-v0', is_slippery=args.slippery)
    V, policy = value_iteration(env, discount_factor=args.discount)
    act_in_environment(env, policy, args.render)
