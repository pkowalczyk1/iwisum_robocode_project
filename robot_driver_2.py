import socket
import json
import numpy as np
import tensorflow as tf
from collections import deque
import random
import os

# TF-Agents imports
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.specs import array_spec
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import epsilon_greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.trajectories import policy_step as ps

# ------------------------------------------------------------
# Hyperparameters
LEARNING_RATE = 1e-3
REPLAY_BUFFER_SIZE = 100000
BATCH_SIZE = 64
TARGET_UPDATE_PERIOD = 500
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
TOTAL_DECAY_BATTLES = 1000
FC_LAYER_PARAMS = (256, 256)  # Network architecture
CHECKPOINT_DIR = 'dqn_checkpoints_3'
# ------------------------------------------------------------

def preprocess_observation(obs_dict):
    """Preprocess Robocode observation into state vector"""
    feature_keys = ["xPos", "yPos", "energy", "enemyDistance", "enemyHeading",
                    "wallHit", "robotHit", "heading"]
    obs = obs_dict["observation"]
    state_list = [
        float(obs.get(k, 0.0)) if obs.get(k, 0.0) is not None else 0.0 for k in feature_keys
    ]
    return np.array(state_list, dtype=np.float32)

class DQNAgent:
    def __init__(self, state_dim, num_actions):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.epsilon_var = tf.Variable(EPSILON_START, dtype=tf.float32)
        self.train_step_counter = tf.Variable(0)

        # Environment specs
        self.obs_spec = array_spec.ArraySpec(
            shape=(state_dim,), dtype=np.float32, name='observation')
        self.action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=num_actions-1, name='action')

        # Q-Network
        self.q_net = q_network.QNetwork(
            self.obs_spec,
            self.action_spec,
            fc_layer_params=FC_LAYER_PARAMS)

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

        # DQN Agent
        self.agent = dqn_agent.DqnAgent(
            ts.time_step_spec(self.obs_spec),
            self.action_spec,
            q_network=self.q_net,
            optimizer=self.optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            gamma=GAMMA,
            target_update_period=TARGET_UPDATE_PERIOD,
            train_step_counter=self.train_step_counter)

        # Policies
        self.greedy_policy = epsilon_greedy_policy.EpsilonGreedyPolicy(self.agent.policy, epsilon=self.epsilon_var)

        # Replay buffer
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=1,
            max_length=REPLAY_BUFFER_SIZE)

        # Checkpointer
        self.checkpoint_dir = CHECKPOINT_DIR
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpointer = common.Checkpointer(
            ckpt_dir=self.checkpoint_dir,
            agent=self.agent,
            max_to_keep=3)

        # Initialize or restore
        self.checkpointer.initialize_or_restore()

    def update_epsilon(self):
        """Decay exploration rate"""
        decay_amount = (EPSILON_START - EPSILON_END) / TOTAL_DECAY_BATTLES
        self.epsilon_var.assign(max(EPSILON_END, self.epsilon_var - decay_amount))
        self.greedy_policy = epsilon_greedy_policy.EpsilonGreedyPolicy(self.agent.policy, epsilon=self.epsilon_var)

    def select_action(self, time_step, force_random=False):
        """Select action using epsilon-greedy policy"""
        # if force_random or random.random() < self.epsilon_var:
        #     action_step = self.random_policy.action(time_step)
        # else:
        #     action_step = self.agent.policy.action(time_step)
        action_step = self.greedy_policy.action(time_step)
        return action_step.action.numpy()[0]

    def store_experience(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        time_step = ts.TimeStep(
            step_type=tf.constant([0 if done else 1], dtype=tf.int32),  # 0=LAST, 1=MID
            reward=tf.constant([reward], dtype=tf.float32),
            discount=tf.constant([GAMMA], dtype=tf.float32),
            observation=tf.constant([state], dtype=tf.float32))

        next_time_step = ts.TimeStep(
            step_type=tf.constant([1 if not done else 2], dtype=tf.int32),  # 2=TERMINAL
            reward=tf.constant([0.0], dtype=tf.float32),
            discount=tf.constant([GAMMA], dtype=tf.float32),
            observation=tf.constant([next_state], dtype=tf.float32))

        action_step = ps.PolicyStep(
            action=tf.constant([action], dtype=tf.int32),
            info=())

        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        self.replay_buffer.add_batch(traj)

    def train(self):
        """Sample from replay buffer and train"""
        if self.replay_buffer.num_frames() < BATCH_SIZE:
            return 0.0

        # Sample batch
        experience, _ = self.replay_buffer.get_next(
            sample_batch_size=BATCH_SIZE,
            num_steps=2)  # Single step transitions

        # Convert to trajectory
        traj = trajectory.Trajectory(
            step_type=experience.step_type,
            observation=experience.observation,
            action=experience.action,
            policy_info=(),
            next_step_type=experience.next_step_type,
            reward=experience.reward,
            discount=experience.discount)

        # Train
        loss_info = self.agent.train(traj)
        return loss_info.loss.numpy()

    def save_checkpoint(self):
        """Save agent state"""
        self.checkpointer.save(self.train_step_counter)
        print(f"Checkpoint saved at step {self.train_step_counter.numpy()}")

def main():
    host = 'localhost'
    port = 5000

    # Initialize agent only once (state_dim will be set after first observation)
    agent = None
    num_actions = 5  # Adjust based on your action space

    # Initialize socket (single persistent connection)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        while True:
            s.listen(1)
            print(f"Listening on {host}:{port}...")

            # Accept single connection that will persist for all battles
            conn, addr = s.accept()
            print(f"Connection established with {addr}")

            with conn:
                input_stream = conn.makefile('r')
                output_stream = conn.makefile('w')

                # Main training loop (single connection)
                while True:
                    # Get initial observation for battle
                    obs_json = input_stream.readline().strip()
                    if not obs_json:
                        print("Connection closed by client")
                        break

                    # Initialize agent on first observation if needed
                    if agent is None:
                        first_obs = json.loads(obs_json)
                        state = preprocess_observation(first_obs)
                        state_dim = len(state)
                        agent = DQNAgent(state_dim, num_actions)
                        print(f"Agent initialized with state_dim={state_dim}, actions={num_actions}")

                    # Process the observation we just received
                    obs = json.loads(obs_json)
                    state = preprocess_observation(obs)
                    done = False
                    battle_reward = 0

                    while not done:
                        # Create time_step
                        time_step = ts.TimeStep(
                            step_type=tf.constant([1], dtype=tf.int32),  # MID
                            reward=tf.constant([0.0], dtype=tf.float32),
                            discount=tf.constant([GAMMA], dtype=tf.float32),
                            observation=tf.constant([state], dtype=tf.float32))

                        # Select action
                        action = agent.select_action(time_step)
                        print(f"Action: {action}, Epsilon: {agent.epsilon_var}")

                        # Send action
                        output_stream.write(f"{action}\n")
                        output_stream.flush()

                        # Get next observation
                        next_obs_json = input_stream.readline().strip()
                        if not next_obs_json:
                            break

                        next_obs = json.loads(next_obs_json)
                        # print(next_obs_json)
                        next_state = preprocess_observation(next_obs)
                        reward = float(next_obs.get("reward", 0.0))
                        battle_reward += reward
                        done = bool(next_obs.get("done", False))

                        # Store experience
                        agent.store_experience(state, action, reward, next_state, done)

                        # Train
                        for _ in range(5):
                            loss = agent.train()
                            if loss > 0:
                                print(f"Step: {agent.train_step_counter.numpy()}, Loss: {loss:.4f}")

                        # Update state
                        state = next_state

                    # Battle ended
                    print(f"Battle ended. Total reward: {battle_reward:.2f}")
                    agent.update_epsilon()
                    agent.save_checkpoint()

                    # Write results
                    with open("results_4.csv", "a") as f:
                        f.write(f"{battle_reward}\n")

if __name__ == '__main__':
    main()