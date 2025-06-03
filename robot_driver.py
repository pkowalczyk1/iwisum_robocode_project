import socket
import json
import numpy as np
import tensorflow as tf
from collections import deque

# TF-Agents imports:
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.specs import array_spec
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import policy_step as ps

# ------------------------------------------------------------
# Helper: preprocess_observation
def preprocess_observation(obs_dict):
    feature_keys = [
        "action", "xPos", "yPos", "energy", "velocity",
        "enemyDistance", "enemyHeading", "wallHit", "robotHit"
    ]
    obs = obs_dict["observation"]
    state_list = [
        float(obs.get(k, 0.0) if obs.get(k, 0.0) is not None else 0.0)
        for k in feature_keys
    ]
    arr = np.array(state_list, dtype=np.float32)
    return arr.reshape((1, -1))
# ------------------------------------------------------------

# Simple experience tuple
class Experience:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

# Simple Replay Buffer
def make_replay_buffer(size=10000):
    return deque(maxlen=size)

# Sample a minibatch from buffer
def sample_batch(buffer, batch_size=32):
    indices = np.random.choice(len(buffer), size=min(len(buffer), batch_size), replace=False)
    batch = [buffer[i] for i in indices]
    states = np.vstack([b.state for b in batch])
    actions = np.array([b.action for b in batch], dtype=np.int32)
    rewards = np.array([b.reward for b in batch], dtype=np.float32)
    next_states = np.vstack([b.next_state for b in batch])
    dones = np.array([b.done for b in batch], dtype=np.float32)
    return states, actions, rewards, next_states, dones


def main():
    host = 'localhost'
    port = 5000

    replay_buffer = make_replay_buffer(size=10000)
    batch_size = 32

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))

        while True:
            s.listen(1)
            conn, addr = s.accept()
            print("Connection established with", addr)

            with conn:
                input_stream  = conn.makefile('r')
                output_stream = conn.makefile('w')

                first_obs_json = input_stream.readline().strip()
                if not first_obs_json:
                    print("No initial observation received; closing.")
                    break

                first_obs = json.loads(first_obs_json)
                state_np   = preprocess_observation(first_obs)
                state_dim  = state_np.shape[1]
                num_actions = 4

                obs_spec = array_spec.ArraySpec(
                    shape=(state_dim,),
                    dtype=np.float32,
                    name='observation'
                )
                action_spec = array_spec.BoundedArraySpec(
                    shape=(),
                    dtype=np.int32,
                    minimum=0,
                    maximum=num_actions - 1,
                    name='action'
                )
                q_net = q_network.QNetwork(
                    obs_spec,
                    action_spec,
                    fc_layer_params=(100, 100)
                )
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
                train_step = tf.Variable(0)
                agent = dqn_agent.DqnAgent(
                    time_step_spec=ts.time_step_spec(obs_spec),
                    action_spec=action_spec,
                    q_network=q_net,
                    optimizer=optimizer,
                    td_errors_loss_fn=common.element_wise_squared_loss,
                    train_step_counter=train_step,
                )
                agent.initialize()
                checkpoint_dir = "checkpoint"
                ckpt = common.Checkpointer(
                    ckpt_dir=checkpoint_dir,
                    agent=agent
                )
                ckpt.initialize_or_restore()
                print("TF-Agents DQN Agent loaded from (or initialized at)", checkpoint_dir)
                policy = agent.policy

                prev_state_np = state_np
                prev_action = None
                prev_reward = None
                while True:
                    step_type = tf.constant([ts.StepType.MID], dtype=tf.int32)
                    reward    = tf.constant([0.0], dtype=tf.float32)
                    discount  = tf.constant([1.0], dtype=tf.float32)
                    obs_tensor = tf.convert_to_tensor(state_np)
                    time_step_obj = ts.TimeStep(
                        step_type=step_type,
                        reward=reward,
                        discount=discount,
                        observation=obs_tensor
                    )
                    action_step = policy.action(time_step_obj)
                    action = int(action_step.action.numpy()[0])
                    print("Action:", action)
                    output_stream.write(f"{action}\n")
                    output_stream.flush()
                    next_obs_json = input_stream.readline().strip()
                    if not next_obs_json:
                        print("Connection closed by peer.")
                        # Save checkpoint here (overwrite last)
                        ckpt.save(train_step)
                        print(f"Agent state saved to checkpoint: {checkpoint_dir}")
                        break
                    next_obs = json.loads(next_obs_json)
                    print("Received observation:", next_obs)
                    next_state_np = preprocess_observation(next_obs)
                    reward_val = float(next_obs.get("reward", 0.0))
                    done = False  # You can add a real done flag if your env provides it

                    # Add experience to replay buffer
                    if prev_action is not None:
                        exp = Experience(prev_state_np, prev_action, prev_reward, state_np, done)
                        replay_buffer.append(exp)

                        # Train after each step if enough data
                        if len(replay_buffer) >= batch_size:
                            states, actions, rewards, next_states, dones = sample_batch(replay_buffer, batch_size)
                            # Convert to tensors
                            states = tf.convert_to_tensor(states, dtype=tf.float32)
                            actions = tf.convert_to_tensor(actions, dtype=tf.int32)
                            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
                            next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
                            dones = tf.convert_to_tensor(dones, dtype=tf.float32)

                            # Expand dims to add time axis
                            def add_time_dim(x):
                                return tf.expand_dims(x, axis=1)

                            states = add_time_dim(states)
                            actions = add_time_dim(actions)
                            rewards = add_time_dim(rewards)
                            next_states = add_time_dim(next_states)
                            dones = add_time_dim(dones)

                            # Make TimeSteps for [B, T, ...]
                            time_steps = ts.TimeStep(
                                step_type=tf.constant([[ts.StepType.MID]] * batch_size, dtype=tf.int32),
                                reward=rewards,
                                discount=tf.constant([[1.0]] * batch_size, dtype=tf.float32),
                                observation=states
                            )
                            next_time_steps = ts.TimeStep(
                                step_type=tf.constant([[ts.StepType.MID]] * batch_size, dtype=tf.int32),
                                reward=tf.zeros_like(rewards),  # not used
                                discount=tf.constant([[1.0]] * batch_size, dtype=tf.float32),
                                observation=next_states
                            )

                            # PolicyStep shape [B, T, ...]
                            policy_step = ps.PolicyStep(action=actions, info=())

                            exp_traj = trajectory.from_transition(
                                time_step=time_steps,
                                action_step=policy_step,
                                next_time_step=next_time_steps
                            )

                            agent.train(exp_traj)
                    # Store for next step
                    prev_state_np = state_np
                    prev_action = action
                    prev_reward = reward_val
                    state_np = next_state_np

if __name__ == '__main__':
    main()
