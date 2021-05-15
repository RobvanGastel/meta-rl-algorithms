import argparse
import copy
import gym
import numpy as np

"""Implement on the environment https://gym.openai.com/envs/FrozenLake-v0/
"""


def policy_evaluation(env, policy, discount_factor=1.0,
                      theta=1e-3):
    V = np.zeros(env.nS)
    while True:
        delta = 0

        for s in range(env.nS):
            v = copy.deepcopy(V[s])
            V_s = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, _ in env.P[s][a]:
                    V_s += action_prob * prob * \
                        (reward + discount_factor * V[next_state])
            V[s] = V_s

            delta = max(delta, np.abs(v-V[s]))
        if delta < theta:
            break
    return np.array(V)


def policy_improvement(env, policy, V, discount_factor):
    def next_actions(s):
        actions = np.zeros(env.nA)
        for action in range(env.nA):
            for prob, next_state, reward, _ in env.P[s][action]:
                actions[action] += prob * (reward + discount_factor
                                           * V[next_state])
        return actions

    policy_stable = True
    for s in range(env.nS):
        old_action = np.argmax(policy[s])
        action_values = next_actions(s)
        best_action = np.argmax(action_values)

        # Keep the actions sparse
        policy[s] = np.eye(env.nA)[best_action]

        if best_action != old_action:
            policy_stable = False
    return V, policy, policy_stable


def policy_iteration(env, discount_factor=1.0):
    # initial equiprobable policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    iteration = 0
    policy_stable = False

    # Value function initiated in policy evaluation
    while not policy_stable:
        V = policy_evaluation(env, policy, discount_factor)
        V, policy, policy_stable = policy_improvement(
            env, policy, V, discount_factor)
        iteration += 1

        print("policy:", policy)
        print("Value function:", V)
        print("Iteration:", iteration)
    return V, policy


def act_in_environment(env, policy, render=False):
    # Now acts through the policy we found with policy iteration,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Execute policy iteration on the FrozenLake environment.')
    parser.add_argument("--slippery", action='store_true')
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--render", default=True, type=bool)
    args = parser.parse_args()

    # Execute policy iteration
    env = gym.make('FrozenLake-v0', is_slippery=args.slippery)
    V, policy = policy_iteration(env, discount_factor=args.discount)
    act_in_environment(env, policy, args.render)
