import numpy as np
import random

from fl_simulation import run_fl

# this code assumed a certain computer's profile (line 225) 
# Then uses the RL agent to pick the best q and N values 

# STATE

class State:
    def __init__(self, cpu, bandwidth, security_level, loss, delta_loss):
        self.cpu = cpu                        # float 0–1 (normalized CPU usage)
        self.bandwidth = bandwidth            # float 0–1 (normalized bandwidth)
        self.security_level = security_level  # int: 128 / 192 / 256
        self.loss = loss                      # float: current FL loss
        self.delta_loss = delta_loss          # float: loss change vs previous round

    def discretize(self):
        # this portion is what is responsible for converting the state into a Q-table key
        # you cannot use raw floats as a key  - two states that are almost identical would never match 
        # this stage essentially makes it into categories
        # CPU bin
        if self.cpu < 0.3:
            cpu_bin = "low"
        elif self.cpu < 0.7:
            cpu_bin = "medium"
        else:
            cpu_bin = "high"

        # Bandwidth bin
        if self.bandwidth < 0.3:
            bandwidth_bin = "low"
        elif self.bandwidth < 0.7:
            bandwidth_bin = "medium"
        else:
            bandwidth_bin = "high"

        # Loss bin
        if self.loss < 0.3:
            loss_bin = "low"
        else:
            loss_bin = "high"

        # Loss trend
        if self.delta_loss < -0.01:
            trend = "improving"
        elif self.delta_loss > 0.01:
            trend = "worsening"
        else:
            trend = "stable"

        # Security level stays as-is (already discrete)
        # BUG FIX: return statement was missing in original
        return (cpu_bin, bandwidth_bin, self.security_level, loss_bin, trend)

    def to_vector(self):
        return [
            self.cpu,
            self.bandwidth,
            self.security_level,
            self.loss,
            self.delta_loss
        ]


# ACTION SPACE

class ActionSpace:
    # at every step, the agent must pick one of exactly these 3 actions
    # each action is a pair of HE parameters
    def __init__(self):
        self.actions = [
            {"N": 2048, "q": 40},   # Action 0: lightweight, low security
            {"N": 4096, "q": 60},   # Action 1: balanced
            {"N": 8192, "q": 80},   # Action 2: heavyweight, high security
        ]

    def get(self, index):
        return self.actions[index]

    def size(self):
        return len(self.actions)



# Q-TABLE HELPERS

def get_q_values(Q, state, action_space):
    # looks up the current state in the Q-table and returns 3 numbers - one per action 
    # the numbers represent how good each action has been historically in this state. higher = better 
    state_key = state.discretize()
    if state_key not in Q:
        Q[state_key] = np.zeros(action_space.size())
    return Q[state_key]


def choose_action(Q, state, action_space, epsilon):
    # Epsilon-greedy action selection
    q_values = get_q_values(Q, state, action_space)

    if random.random() < epsilon:
        return random.randint(0, action_space.size() - 1)   # explore
    else:
        return int(np.argmax(q_values))                     # exploit
    # as training goes on, epsilon (probability of exploration) goes down 


# ENVIRONMENT STEP

def security_score(N):
    
    # Map N to a normalized security score [0, 1].
    # Higher N = stronger security per CKKS recommendations.
    
    scores = {2048: 0.0, 4096: 0.5, 8192: 1.0}
    return scores.get(N, 0.5)


def step(Q, state, action, action_space):
    """
    This is the part that actually does reinforcement learning.

    Apply the chosen HE parameters, run one FL round, and compute reward.

    Reward design follows the HERL paper:
        reward = - (utility_cost + latency_cost + comm_cost - security_bonus)

    where:
        utility_cost  = weight on FL loss (we want low loss)
        latency_cost  = compute overhead proxy (scales with N and q)
        comm_cost     = communication overhead proxy (scales with q / bandwidth)
        security_bonus= reward for picking stronger encryption when required

    A high security_level state gets penalized for choosing weak encryption
    (low N), matching the HERL paper's security-aware reward function.
    """
    N = action["N"]
    q = action["q"]

    # Cost terms (defined by me)
    time_cost  = ((N / 1000) + (q / 10)) / (state.cpu + 1e-5) #1e-5 term added to ensure no error if state.cpu = 0 
    comm_cost  = (q / 10) / (state.bandwidth + 1e-5)

    # Run FL with chosen HE params 
    new_loss   = run_fl(N, q) # imported function from fl_simulation.py
    delta_loss = new_loss - state.loss

    # Security bonus/penalty
    # If client needs high security (256) but agent picks weak params, penalize
    # Normalize security_level to [0, 1]: 128->0.0, 192->0.5, 256->1.0
    required_security = (state.security_level - 128) / 128.0
    achieved_security = security_score(N)
    security_mismatch = max(0.0, required_security - achieved_security)
   
    # Final reward - we first normalize all the terms otherwise time cost dominates 
    utility_term  = new_loss / 2.5
    latency_term  = time_cost / 27.0
    comm_term     = comm_cost / 20.0
    security_term = security_mismatch       # already 0–1


    reward = -(0.4 * utility_term  +
        0.3 * latency_term  +
        0.2 * comm_term     +
        0.1 * security_term)

    # Next state (cpu/bandwidth/security unchanged in simulation)
    next_state = State(
        cpu=state.cpu,
        bandwidth=state.bandwidth,
        security_level=state.security_level,
        loss=new_loss,
        delta_loss=delta_loss
    )

    return next_state, reward



# Q-LEARNING UPDATE

def update_q(Q, state, action_idx, reward, next_state, action_space,
             alpha=0.1, gamma=0.9):
    # Bellman update for the Q-table
    # let's say I'd initially given a score of 6 now i have rated it a 7, instead of fully wiping the 6 and replacing with the 7
    # this just nudges the updates in the direction of 7
    # the future updates are sort of a 'preview' of future consequences, baked into every single update

    s      = state.discretize()
    s_next = next_state.discretize()

    if s not in Q:
        Q[s] = np.zeros(action_space.size())
    if s_next not in Q:
        Q[s_next] = np.zeros(action_space.size())

    best_next = np.max(Q[s_next])
    Q[s][action_idx] += alpha * (reward + gamma * best_next - Q[s][action_idx])


# -----------------------------
# MAIN TRAINING LOOP
# -----------------------------
if __name__ == "__main__": # runs only when you directly call rl_agent.py

    action_space = ActionSpace()
    Q = {}

    # Hyperparameters
    alpha         = 0.1    # learning rate
    gamma         = 0.9    # discount factor for future updates
    epsilon       = 1.0    # starting exploration rate
    epsilon_min   = 0.05   # minimum exploration rate
    epsilon_decay = 0.85   # multiplicative decay per episode

    num_episodes  = 10     # kept small as FL+HE is slow
    steps_per_ep  = 5

    for episode in range(num_episodes):

        # Reset state at start of each episode to the same starting point
        state = State( #assumed client profile
            cpu=0.6,
            bandwidth=0.4,
            security_level=128,
            loss=0.5,
            delta_loss=0.0
        )

        episode_reward = 0.0

        for t in range(steps_per_ep):
            # Choose action
            action_idx = choose_action(Q, state, action_space, epsilon)
            action     = action_space.get(action_idx)

            # Step environment : run FL with given params, get reward and next state 
            next_state, reward = step(Q, state, action, action_space)
            episode_reward += reward

            print(f"  [Ep {episode} | Step {t}] "
                  f"Action: N={action['N']} q={action['q']} | "
                  f"Loss: {next_state.loss:.4f} | "
                  f"Reward: {reward:.4f}")

            # Q-table update using Bellman equation 
            update_q(Q, state, action_idx, reward, next_state,
                     action_space, alpha=alpha, gamma=gamma)

            state = next_state

        # Decay epsilon after each episode
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(f"Episode {episode} done | "
              f"Total reward: {episode_reward:.4f} | "
              f"Epsilon: {epsilon:.3f}\n")

    # Print final Q-table
    print("=== Final Q-Table ===")
    for state_key, q_vals in Q.items():
        best_action = int(np.argmax(q_vals))
        print(f"  State {state_key}")
        print(f"    Q-values: {np.round(q_vals, 4)}")
        print(f"    Best action: {action_space.get(best_action)}")