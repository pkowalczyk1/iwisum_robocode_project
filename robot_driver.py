import socket
import json
import numpy as np
import tensorflow as tf

# TF-Agents imports:
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.specs import array_spec
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts

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

def main():
    host = 'localhost'
    port = 5000

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
                # Try to restore from last checkpoint, if available
                ckpt.initialize_or_restore()
                print("TF-Agents DQN Agent loaded from (or initialized at)", checkpoint_dir)
                policy = agent.policy

                while True:
                    step_type = tf.constant([ts.StepType.MID], dtype=tf.int32)
                    reward    = tf.constant([0.0], dtype=tf.float32)
                    discount  = tf.constant([1.0], dtype=tf.float32)
                    obs_tensor = tf.convert_to_tensor(state_np)
                    time_step = ts.TimeStep(
                        step_type=step_type,
                        reward=reward,
                        discount=discount,
                        observation=obs_tensor
                    )
                    action_step = policy.action(time_step)
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
                    state_np = preprocess_observation(next_obs)

if __name__ == '__main__':
    main()
