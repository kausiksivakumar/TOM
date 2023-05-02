import argparse
import logging
import time
import gym
import d4rl
import torch
# torch.set_default_dtype(torch.float64)
import numpy as np
from itertools import count
from sac.replay_memory import ReplayMemory
from sac.sac import SAC
from ensemble import ProbEnsemble
from predict_env import PredictEnv
from sample_env import EnvSampler
# from tf_models.constructor import construct_model, format_samples_for_training
# torch.autograd.set_detect_anomaly(True)
# import wandb
from tqdm import tqdm
from utils import *

def readParser():
    parser = argparse.ArgumentParser(description='SAC')

    parser.add_argument('--env', default="hopper-medium-expert-v0",
                        help='Mujoco Gym environment (default: hopper-medium-expert-v0)')
    parser.add_argument('--algo', default="sac",
                        help='Must be sac')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
        help='random seed (default: 123456)')

    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
    parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')

    parser.add_argument('--num_networks', type=int, default=7, metavar='E',
                    help='ensemble size (default: 7)')
    parser.add_argument('--num_elites', type=int, default=5, metavar='E',
                    help='elite size (default: 5)')
    parser.add_argument('--pred_hidden_size', type=int, default=200, metavar='E',
                    help='hidden size for predictive model')
    parser.add_argument('--reward_size', type=int, default=1, metavar='E',
                    help='environment reward size')

    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
    parser.add_argument('--epoch_length', type=int, default=1000, metavar='A',
                    help='steps per epoch')

    parser.add_argument('--num_epoch', type=int, default=1000, metavar='A',
                    help='total number of epochs')
    parser.add_argument('--min_pool_size', type=int, default=1000, metavar='A',
                    help='minimum pool size')
    parser.add_argument('--real_ratio', type=float, default=1, metavar='A',
                    help='ratio of env samples / model samples')
    parser.add_argument('--train_every_n_steps', type=int, default=1, metavar='A',
                    help='frequency of training policy')
    parser.add_argument('--num_train_repeat', type=int, default=1, metavar='A',
                    help='times to training policy per step')
    parser.add_argument('--eval_n_episodes', type=int, default=10, metavar='A',
                    help='number of evaluation episodes')
    parser.add_argument('--max_train_repeat_per_step', type=int, default=5, metavar='A',
                    help='max training times per step')
    parser.add_argument('--policy_train_batch_size', type=int, default=256, metavar='A',
                    help='batch size for training policy')
    parser.add_argument('--init_exploration_steps', type=int, default=5000, metavar='A',
                    help='exploration steps initially')

    parser.add_argument('--model_type', default='pytorch', metavar='A',
                    help='predict model -- pytorch or tensorflow')

    parser.add_argument('--cuda', default=True, action="store_true",
                    help='run on CUDA (default: True)')
    return parser.parse_args()


def train(args, env_sampler, predict_env, agent, env_pool, model_pool):
    total_step = 0
    reward_sum = 0
    rollout_length = 1
    exploration_before_start(args, env_sampler, env_pool, agent)

    for epoch_step in tqdm(range(args.num_epoch)):
        if epoch_step % 100 == 0:
            buffer_path = f'dataset/{args.env}-{args.algo}-epoch{epoch_step}.npy'
            env_pool.save_buffer(buffer_path)
            agent_path = f'saved_policies/{args.env}-{args.algo}-epoch{epoch_step}'
            agent.save_model(agent_path)

        start_step = total_step
        train_policy_steps = 0
        for i in count():
            cur_step = total_step - start_step

            # epoch_length = 1000, min_pool_size = 1000
            if cur_step >= args.epoch_length and len(env_pool) > args.min_pool_size:
                break

            # step in real environment
            cur_state, action, next_state, reward, done, info = env_sampler.sample(agent)
            env_pool.push(cur_state, action, reward, next_state, done)

            # train policy
            if len(env_pool) > args.min_pool_size:
                train_policy_steps += train_policy_repeats(args, total_step, train_policy_steps, cur_step, env_pool, model_pool, agent)

            total_step += 1

        rewards = [evaluate_policy(env_sampler, agent, args.epoch_length) for _ in range(args.eval_n_episodes)]

        print("")
        print(f'Epoch {epoch_step} Eval_Reward {np.mean(rewards)} Eval_Std {np.std(rewards)}')
        # wandb.log({'eval_reward': np.mean(rewards),
        #            'eval_std': np.std(rewards)})


def evaluate_policy(env_sampler, agent, epoch_length=1000):
    env_sampler.current_state = None
    sum_reward = 0
    for t in range(epoch_length):
        cur_state, action, next_state, reward, done, info = env_sampler.sample(agent, eval_t=True)
        sum_reward += reward
        if done:
            break
    return sum_reward

def exploration_before_start(args, env_sampler, env_pool, agent):
    # init_exploration_steps = 5000
    for i in range(args.init_exploration_steps):
        cur_state, action, next_state, reward, done, info = env_sampler.sample(agent)
        env_pool.push(cur_state, action, reward, next_state, done)

def train_policy_repeats(args, total_step, train_step, cur_step, env_pool, model_pool, agent):
    # train_every_n_steps: 1
    if total_step % args.train_every_n_steps > 0:
        return 0
    # max_train_repeat_per_step: 5
    if train_step > args.max_train_repeat_per_step * total_step:
        return 0

    # num_train_repeat: 20
    for i in range(args.num_train_repeat):
        env_batch_size = int(args.policy_train_batch_size * args.real_ratio)
        model_batch_size = args.policy_train_batch_size - env_batch_size

        env_state, env_action, env_reward, env_next_state, env_done = env_pool.sample(int(env_batch_size))

        if model_batch_size > 0 and len(model_pool) > 0:
            model_state, model_action, model_reward, model_next_state, model_done = model_pool.sample_all_batch(int(model_batch_size))
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = np.concatenate((env_state, model_state), axis=0), \
                np.concatenate((env_action, model_action), axis=0), np.concatenate((np.reshape(env_reward, (env_reward.shape[0], -1)), model_reward), axis=0), \
                np.concatenate((env_next_state, model_next_state), axis=0), np.concatenate((np.reshape(env_done, (env_done.shape[0], -1)), model_done), axis=0)
        else:
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = env_state, env_action, env_reward, env_next_state, env_done

        batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
        batch_done = (~batch_done).astype(int)

        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters((batch_state, batch_action, batch_reward, batch_next_state, batch_done), args.policy_train_batch_size, i)

        # wandb.log({'critic1_loss': critic_1_loss,
        #            'critic2_loss': critic_2_loss,
        #            'policy_loss': policy_loss,
        #            'entropy_loss': ent_loss,
        #            'alpha': alpha})

    return args.num_train_repeat


def main():
    args = readParser()

    # Initial environment
    if args.env == 'Ant-v2':
        from env.ant import AntTruncatedObsEnv
        env = AntTruncatedObsEnv()
    elif args.env == 'Humanoid-v2':
        from env.humanoid import HumanoidTruncatedObsEnv
        env = HumanoidTruncatedObsEnv()
        args.automatic_entropy_tuning = True

    else:
        env = gym.make(args.env)

    # wandb.init(project='mdn-mbrl',
    #            group=args.env.split('-')[0],
    #            name=f"{args.algo}-{args.seed}",
    #            config=args)

    # env = gym.make(args.env)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    # Intial agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    # hack to check humanoid is working
    # if args.env == 'Humanoid-v2':
    #     agent.target_entropy = -2
    #     agent.alpha = 0.05


    # Initial ensemble model
    state_size = np.prod(env.observation_space.shape)
    action_size = np.prod(env.action_space.shape)
    if args.model_type == 'pytorch':
        # env_model = Ensemble_Model(args.num_networks, args.num_elites, state_size, action_size, args.reward_size, args.pred_hidden_size)
        env_model = ProbEnsemble(args.num_networks, args.num_elites, state_size, action_size, args.reward_size, args.pred_hidden_size)
    else:
        env_model = construct_model(obs_dim=state_size, act_dim=action_size, hidden_dim=args.pred_hidden_size, num_networks=args.num_networks, num_elites=args.num_elites)

    if args.cuda:
        env_model.to('cuda')

    # Predict environments
    predict_env = PredictEnv(env_model, args.env, args.model_type)

    # Initial pool for env
    env_pool = ReplayMemory(args.replay_size)

    # Initial pool for model
    model_pool = ReplayMemory(1)

    # Sampler of environment
    env_sampler = EnvSampler(env)

    train(args, env_sampler, predict_env, agent, env_pool, model_pool)


if __name__ == '__main__':
    main()
