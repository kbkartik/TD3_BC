import numpy as np
import torch
import gym
import argparse
import os
import d4rl

import utils
import TD3_BC
import csv
import datetime


# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
print('Threads: ', torch.get_num_threads())
def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + seed_offset)

	avg_reward = 0.
	pen0 = pen1 = pen2 = pen3 = pen4 = pen5 = 0
	act_sum = unsafe_height = unsafe_ang = 0
	done_count = steps = total_unsafe = 0
	avg_act_sum = avg_done_count = avg_step_rew = avg_steps = 0.0
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			state = (np.array(state).reshape(1,-1) - mean)/std
			action = policy.select_action(state)
			state, reward, done, _ = eval_env.step(action)
			
			if action[0] > 0.9 or action[0] < -0.9: 
				pen0 += 1
			if action[1] > 0.9 or action[1] < -0.9:
				pen1 += 1
			if action[2] > 0.9 or  action[2] < -0.9:
				pen2 += 1
			if action[3] > 0.9 or action[3] < -0.9:
				pen3 += 1
			if action[4] > 0.9 or action[4] < -0.9:
				pen4 += 1
			if action[5] > 0.9 or  action[5] < -0.9:
				pen5 += 1
			if state[0] < 0.8 or state[0] > 2.0:
				unsafe_height += 1
			if abs(state[1]) > 1.0:
				unsafe_ang += 1
			if done == True:
				done_count += 1
			
			steps += 1
			act_sum += action[0]+action[1]+action[2]+action[3]+action[4]+action[5]
			avg_reward += reward
	
	total_unsafe = unsafe_ang + unsafe_height
	avg_step_rew = avg_reward / steps
	avg_reward /= eval_episodes
	avg_act_sum = act_sum/eval_episodes
	avg_done_count = done_count/eval_episodes
	avg_steps = steps/eval_episodes
	
	d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}, unsafe states: {total_unsafe, int(act_sum), avg_step_rew}")
	print("---------------------------------------")

	return [d4rl_score, avg_reward, pen0, pen1, pen2, pen3, pen4, pen5, int(act_sum), steps, avg_steps, unsafe_height, unsafe_ang, total_unsafe, avg_act_sum, done_count, avg_step_rew]


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	# Experiment
	parser.add_argument("--policy", default="TD3_BC")               # Policy name
	parser.add_argument("--env", default="hopper-medium-v2")        # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--save_model", default=False)        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	# TD3
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	# TD3 + BC
	parser.add_argument("--alpha", default=2.5)
	parser.add_argument("--normalize", default=True)
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		# TD3
		"policy_noise": args.policy_noise * max_action,
		"noise_clip": args.noise_clip * max_action,
		"policy_freq": args.policy_freq,
		# TD3 + BC
		"alpha": args.alpha
	}

	# Initialize policy
	policy = TD3_BC.TD3_BC(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
	if args.normalize:
		mean,std = replay_buffer.normalize_states() 
	else:
		mean,std = 0,1
	
	evaluations = []
	start_time = datetime.datetime.now()
	save_csv = []
	fields = ['D4RL Score', 'Avg Reward', 'action0_a', 'action0_b', 'action1_a', 'action1_b', 'action2_a', 'action2_b', 'act_sum', 'steps', 'avg_steps', 'unsafe_height', 'unsafe_ang', 'total_unsafe',  'avg_act_sum', 'done_count', 'avg_step_rew']
	for t in range(int(args.max_timesteps)):
		policy.train(replay_buffer, args.batch_size)
		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			print(f"Time steps: {t+1}")
			save_data = eval_policy(policy, args.env, args.seed, mean, std)
			save_csv.append(save_data)
			if args.save_model: policy.save(f"./models/{file_name}")

	with open('./results/'+args.env+'_seed_'+str(args.seed)+'.csv', 'w') as f:
		# using csv.writer method from CSV package
		write = csv.writer(f)
		write.writerow(fields)
		write.writerows(save_csv)
	
	#if torch.cuda.get_device_name(torch.device) != 'cpu':
	#	torch.cuda.synchronize()
	end_time = datetime.datetime.now()
	print('Total training time: ', end_time - start_time)