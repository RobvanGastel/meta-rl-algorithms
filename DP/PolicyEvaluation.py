import copy
import numpy as np
from gridworld import GridworldEnv


"""Original implementation by
https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/PolicyEvaluation.py
"""


env = GridworldEnv()


def policy_evaluation(policy, env, discount_factor=1.0, theta=1e-3):
    """
    Iteratively evaluate a policy given an environment and a full 
    description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities
        of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state,
            reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less
        than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        delta = 0

        # Loop for each s in S:
        for s in range(env.nS):
            v = copy.deepcopy(V[s])
            V_s = 0
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for prob, next_state, reward, _ in env.P[s][a]:
                    # Calculate the expected value. Ref: Sutton book eq. 4.6.
                    V_s += action_prob * prob * \
                        (reward + discount_factor * V[next_state])

                    # Or directly use the transition dynamics.
                    # V_s += policy[s, a] * env.P[s][a][0][0] * \
                    #     (env.P[s][a][0][2] + discount_factor * V[env.P[s][a][0][1]])
            V[s] = V_s

            delta = max(delta, np.abs(v-V[s]))
        if delta < theta:
            break
    return np.array(V)


random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_evaluation(random_policy, env)

# Test: Make sure the evaluated policy is what we expected
expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -
                       20, -20, -20, -18, -14, -22, -20, -14, 0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
