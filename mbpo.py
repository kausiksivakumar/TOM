import argparse
from email import policy
import time
import gym
import torch
import numpy as np
from itertools import count, permutations
from sac.replay_memory import ReplayMemory
from sac.sac import SAC
from mdn import MDN
from predict_env import PredictEnv
from sample_env import EnvSampler
from smodice_pytorch import SMODICE_TOM
from vaml import VAML # Kausik : check this 
# from tf_models.constructor import construct_model, format_samples_for_training
# torch.autograd.set_detect_anomaly(True)
# import wandb
from tqdm import tqdm
from discriminator_pytorch import Discriminator_SAS
from utils import create_data_loader, permute_and_pass, sample_init_state, cal_weights, weight_vs_position, update_litm_prob, get_weights_pos_bar,save_policy_gif, model_error, truncated_linear
import copy
def readParser():
    parser = argparse.ArgumentParser(description='MBPO')
    parser.add_argument('--env', default="Hopper-v2",
        help='Mujoco Gym environment (default: Hopper-v2)')
    parser.add_argument('--model', default='mdn', metavar='A',
                    help='predict model -- ensemble or mdn')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
        help='random seed (default: 123456)')

    # SAC Hyperparameters
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

    # Ensemble Model Hyperparameters
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
    parser.add_argument('--model_retain_epochs', type=int, default=1, metavar='A',
                    help='retain epochs')
    parser.add_argument('--model_train_freq', type=int, default=250, metavar='A',
                    help='frequency of training')
    parser.add_argument('--rollout_batch_size', type=int, default=100000, metavar='A',
                    help='rollout number M')
    parser.add_argument('--epoch_length', type=int, default=1000, metavar='A',
                    help='steps per epoch')
    parser.add_argument('--rollout_length', type=int, default=1)

    # parser.add_argument('--rollout_schedule', type=list, default=[])
    # parser.add_argument('--rollout_min_epoch', type=int, default=10, metavar='A',
    #                 help='rollout min epoch')
    # parser.add_argument('--rollout_max_epoch', type=int, default=100, metavar='A',
    #                 help='rollout max epoch')
    # parser.add_argument('--rollout_min_length', type=int, default=1, metavar='A',
    #                 help='rollout min length')
    # parser.add_argument('--rollout_max_length', type=int, default=15, metavar='A',
    #                 help='rollout max length')
    # parser.add_argument('--adaptive_rollout', default=False, action="store_true")

    parser.add_argument('--num_epoch', type=int, default=200,   metavar='A',
                    help='total number of epochs')
    parser.add_argument('--min_pool_size', type=int, default=1000, metavar='A',
                    help='minimum pool size')
    parser.add_argument('--real_ratio', type=float, default=0.05, metavar='A',
                    help='ratio of env samples / model samples')
    parser.add_argument('--train_every_n_steps', type=int, default=1, metavar='A',
                    help='frequency of training policy')
    parser.add_argument('--num_train_repeat', type=int, default=20, metavar='A',
                    help='times to training policy per step')
    parser.add_argument('--eval_n_episodes', type=int, default=10, metavar='A',
                    help='number of evaluation episodes')
    parser.add_argument('--max_train_repeat_per_step', type=int, default=5, metavar='A',
                    help='max training times per step')
    parser.add_argument('--policy_train_batch_size', type=int, default=256, metavar='A',
                    help='batch size for training policy')
    parser.add_argument('--init_exploration_steps', type=int, default=5000, metavar='A',
                    help='exploration steps initially')

    # TOM parameters
    parser.add_argument('--method', type=str, default='tom', metavar='N',
                        help='What to run TOM or MBPO? default:TOM')
    parser.add_argument('--disc_hidden', type=int, default=256, metavar='D',
                        help='Discriminator hidden size')
    parser.add_argument('--disc_iter', type=int, default=int(100), metavar='D',
                        help='Discriminator iterations to train')
    parser.add_argument('--q_iter', type=int, default=int(1000), metavar='D',
                        help='Q iterations to train')
    parser.add_argument('--d_batch_size', type=int, default=int(256), metavar='d',
                        help='discriminator batch size')
    parser.add_argument('--m_batch_size', type=int, default=int(256), metavar='M',
                        help='model learning batch size')
    parser.add_argument('--policy_pool_size', type=int, default=1000, metavar='N',
                        help='size of current policy buffer (default: 10000)')
    parser.add_argument('--f', default='chi', type=str, help="Type of f divergence used")
    parser.add_argument('--hidden_sizes', default=(256, 256),metavar='Q',
                        help="Hidden size for TOM-Q network")
    parser.add_argument('--q_l2_reg', default=0.0001, type=float,help="l2 reg param for TOM-Q learning")
    parser.add_argument('--max_epochs', type=int, default=30, metavar='D',
                        help='max epochs to iterate through the ensemble model - default 30 for MDN')
    

    # Debug arguments
    # parser.add_argument('--use_disc', type=str, default=True, metavar='D',
    #                     help='Train Discriminator? Default:True else use Binary rewards')
    parser.add_argument('--use_disc', action='store_true')
    parser.add_argument('--no_disc', dest='feature', action='store_false')
    parser.set_defaults(feature=True)


    return parser.parse_args()


def train(args, env_sampler, predict_env, agent, env_pool, model_pool, cur_pol_pool, tom, disc,device,debug_sampler):
    total_step = 0
    reward_sum = 0
    rollout_length = args.rollout_length
    exploration_before_start(args, env_sampler, env_pool, agent)

    save_gif_step   =   np.linspace(0,args.num_epoch-1,3,dtype = int)
    # Populate probablities if LITM is the method
    if("litm" in args.method):
        prob            =   np.ones((len(env_pool),))/len(env_pool)
        env_pool.prob   =   prob

    train_steps         =   0
    for epoch_step in tqdm(range(args.num_epoch)):
        start_step          =   total_step
        train_policy_steps  =   0
        for i in range(args.epoch_length):
            cur_step = total_step - start_step

            # epoch_length = 1000, min_pool_size = 1000
            if cur_step >= args.epoch_length and len(env_pool) > args.min_pool_size:
                break

            if cur_step % args.model_train_freq == 0 and args.real_ratio < 1.0:
                # train ensemble
             
                # TODO: Check if method is TOM and current policy pool is updated for atleast 1k transitions
                if(args.method=="tom" and len(cur_pol_pool) >= 1000):
                    
                    env_samples, env_loader, cur_samples, cur_loader        =   create_data_loader(env_pool, cur_pol_pool,batch_size=args.d_batch_size)
                    
                    # TODO: Train Discriminator, if not used go with binary rewards
                    if(args.use_disc):
                        print("Training Discriminator")
                        disc                    =   Discriminator_SAS(env_samples[0].shape[1], env_samples[1].shape[1], hidden_dim=args.disc_hidden, device=device).to(device)
                        for itr in tqdm(range(args.disc_iter)):
                            loss                =   disc.update(cur_loader, env_loader)        

                    # TODO: Train TOM network - Interleaving Q update and model learning like SMODICE   
                    print("Training Q network - TOM")
                    th_state,th_action,th_reward,th_next_state,th_terminal,th_disc_reward   =   permute_and_pass(env_samples)
                    permutation                                                             =   np.random.choice(th_state.shape[0],size = args.q_iter*args.m_batch_size)
                    
                    # For fair comparison with baseline, feeding same number of samples (i.e) for 30 epochs over the entire env_pool
                    for u in tqdm(range(0,args.q_iter*args.m_batch_size,args.m_batch_size)):
                        idxs                                                =   permutation[u:u+args.m_batch_size]
                        state,action,reward,next_state,terminal,disc_reward =   th_state[idxs],th_action[idxs],th_reward[idxs],th_next_state[idxs],th_terminal[idxs],th_disc_reward[idxs]                                             
                        
                        # TODO: Get initial state
                        init_state                                          =   sample_init_state(env_pool,args.m_batch_size)   

                        # TODO:Calculate discriminator reward if needed
                        if(args.use_disc):
                            with torch.no_grad():
                                disc_input                                  =   torch.cat([state, action,next_state], axis=1)
                                disc_reward                                 =   disc.predict_reward(disc_input)

                        train_loss                                          =   tom.train_step(init_state, state, action, disc_reward, reward, next_state, terminal, pred_env=predict_env, policy=agent)
                    
                    # TODO: Train dynamics model
                    print("Training dynamics model")
                    # TODO: Calculate disc reward if needed
                    if(args.use_disc):
                        with torch.no_grad():
                            disc_input                                      =   torch.cat([th_state, th_action, th_next_state], axis=1)
                            th_disc_reward                                  =   disc.predict_reward(disc_input)
                    # TODO: Find weights and calculate sample based regression 
                    w_e                                                     =   tom.train_model_step(th_state, th_action, th_disc_reward,th_reward, th_next_state,th_terminal, pred_env=predict_env, policy=agent) 
                
                # TODO: VAML training
                elif "vaml" in args.method:
                    predict_env.model.set_gradient_buffer(args,env_sampler.env.observation_space.shape)
                    predict_env.model.set_agent(agent)
                    train_predict_model(args, env_pool, predict_env)
                
                # TODO: Else it is either LITM or MBPO
                else:
                    # TODO -- sanity check, weights should be None, if litm is not the method
                    train_predict_model(args, env_pool, predict_env)

                start = time.time()
                rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length,tom = tom, disc = disc)
                print("Rollout time --- %s seconds ---" % (time.time() - start))

            # step in real environment
            prev_env_length           = len(env_pool)
            cur_state, action, next_state, reward, done, info = env_sampler.sample(agent)
            env_pool.push(cur_state, action, reward, next_state, done)
            if(env_sampler.path_length==1):
                env_pool.init_state.append(np.array(cur_state))
            cur_pol_pool.push(cur_state, action, reward, next_state, done)            
            # TODO: if method is litm, update its weights
            if("litm" in args.method):
                update_litm_prob(env_pool,train_steps,0.9,prev_env_length)

            # train policy
            if len(env_pool) > args.min_pool_size:
                train_steps         = train_policy_repeats(args, total_step, train_policy_steps, cur_step, env_pool, model_pool, agent,epoch_step)
                train_policy_steps += train_steps
            total_step += 1

        to_log                      =   {}
        rewards                     =   [evaluate_policy(env_sampler, agent, args.epoch_length) for _ in range(args.eval_n_episodes)]
        # pred_err    =   np.array([model_pred_error(debug_sampler,predict_env,agent,epoch_length=2) for _ in range(5)])
       
        print("")
        print(f'Epoch {epoch_step} Eval_Reward {np.mean(rewards)} Eval_Std {np.std(rewards)}')
        # TODO: plot one step model prediction error
        debug_sampler   =   copy.deepcopy(env_sampler)
        to_log.update({'epoch': epoch_step,
                        'eval_reward': np.mean(rewards),
                        'eval_std': np.std(rewards)})
        
        if "vaml" in args.method:
            predict_env.model.add_mse   =   False

        if("tom" in args.method):
            to_log.update(weight_vs_position(env_pool,disc,tom,agent,use_disc=args.use_disc,buffer_size=1000))      
            
        print(f"epoch: {epoch_step}, eval_reward: {to_log['eval_reward']}, eval_std: {to_log['eval_std']}")

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
        
        if(env_sampler.path_length==1):
            env_pool.init_state.append(np.array(cur_state))

def set_rollout_length(args, epoch_step):
    rollout_length = (min(max(args.rollout_min_length + (epoch_step - args.rollout_min_epoch)
        / (args.rollout_max_epoch - args.rollout_min_epoch) * (args.rollout_max_length - args.rollout_min_length),
        args.rollout_min_length), args.rollout_max_length))
    return int(rollout_length)


def train_predict_model(args, env_pool, predict_env):
    # Get all samples from environment
    state, action, reward, next_state, done,_,weights = env_pool.sample(len(env_pool))
    delta_state = next_state - state
    inputs = np.concatenate((state, action), axis=-1)
    labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1)

    # max_epochs = 30
    # if 'mdn' in args.model:
    #     args.max_epochs = 30
    if weights is not None:
        weights = weights.reshape((-1,1))
    if 'vaml' in args.method:
        val_mse, val_nll = predict_env.model.train(inputs, labels, next_state, batch_size=args.m_batch_size, max_epochs=args.max_epochs, weights=weights)
    else:
        val_mse, val_nll = predict_env.model.train(inputs, labels, batch_size=args.m_batch_size, max_epochs=args.max_epochs, weights=weights)

    # wandb.log({'model_nll': val_nll,
    #            'model_rmse': val_mse})

def resize_model_pool(args, rollout_length, model_pool):
    rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
    model_steps_per_epoch = int(rollout_length * rollouts_per_epoch)
    new_pool_size = args.model_retain_epochs * model_steps_per_epoch
    sample_all = model_pool.return_all()
    new_model_pool = ReplayMemory(new_pool_size)
    new_model_pool.push_batch(sample_all)

    return new_model_pool


def rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length, tom = None, disc = None):
    # TODO: Sample init state according to importance weights
    if("tom" in args.method):
        if(tom is None or disc is None or policy is None):
            raise Exception("You either haven't provided the tom object or discriminator obj or policy obj in rollout model")

        state, action, reward, next_state, done, disc_reward,_  =   env_pool.sample(len(env_pool))
        w_e                                                     =   cal_weights(state,action,reward,next_state,done,disc_reward,disc,tom,agent,use_disc = args.use_disc)
        prob                                                    =   (w_e/w_e.sum()).flatten()
        idx                                                     =   np.random.choice(state.shape[0],args.rollout_batch_size,p=(prob)/(prob).sum())
        state                                                   =   state[idx] 
    elif("litm" in args.method):
        state, action, reward, next_state, done = env_pool.sample_all_batch(args.rollout_batch_size,litm=True)
    else:
        state, action, reward, next_state, done = env_pool.sample_all_batch(args.rollout_batch_size)
    for i in range(rollout_length):
        # TODO: Get a batch of actions
        action = agent.select_action(state)
        next_states, rewards, terminals, info = predict_env.step(state, action)
        # TODO: Push a batch of samples
        model_pool.push_batch([(state[j], action[j], rewards[j], next_states[j], terminals[j]) for j in range(state.shape[0])])
        nonterm_mask = ~terminals.squeeze(-1)
        if nonterm_mask.sum() == 0:
            break
        state = next_states[nonterm_mask]


def train_policy_repeats(args, total_step, train_step, cur_step, env_pool, model_pool, agent, epoch_step):
    # train_every_n_steps: 1
    if total_step % args.train_every_n_steps > 0:
        return 0
    # max_train_repeat_per_step: 5
    if train_step > args.max_train_repeat_per_step * cur_step:
        return 0

    # num_train_repeat: 20
    for i in range(args.num_train_repeat):
        env_batch_size = int(args.policy_train_batch_size * args.real_ratio)
        model_batch_size = args.policy_train_batch_size - env_batch_size

        env_state, env_action, env_reward, env_next_state, env_done,_,_ = env_pool.sample(int(env_batch_size))
        if "vaml" in args.method:
            model_data_likelihood   =   truncated_linear(0,20,0,0.95,epoch_step)
            buffer_choice = np.random.choice([True, False], p=[model_data_likelihood, 1.-model_data_likelihood])
            if buffer_choice:
                model_state, model_action, model_reward, model_next_state, model_done = model_pool.sample_all_batch(int(model_batch_size))
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = np.concatenate((env_state, model_state), axis=0), \
                    np.concatenate((env_action, model_action), axis=0), np.concatenate((np.reshape(env_reward, (env_reward.shape[0], -1)), model_reward), axis=0), \
                    np.concatenate((env_next_state, model_next_state), axis=0), np.concatenate((np.reshape(env_done, (env_done.shape[0], -1)), model_done), axis=0)
            else:
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = env_state, env_action, env_reward, env_next_state, env_done

        else:    
            if model_batch_size > 0 and len(model_pool) > 0:
                model_state, model_action, model_reward, model_next_state, model_done = model_pool.sample_all_batch(int(model_batch_size))
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = np.concatenate((env_state, model_state), axis=0), \
                    np.concatenate((env_action, model_action), axis=0), np.concatenate((np.reshape(env_reward, (env_reward.shape[0], -1)), model_reward), axis=0), \
                    np.concatenate((env_next_state, model_next_state), axis=0), np.concatenate((np.reshape(env_done, (env_done.shape[0], -1)), model_done), axis=0)
            else:
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = env_state, env_action, env_reward, env_next_state, env_done

        batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
        # batch_mask = 1 - batch_done
        batch_mask = (~batch_done).astype(int)

        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters((batch_state, batch_action, batch_reward, batch_next_state, batch_mask), args.policy_train_batch_size, i)

    # wandb.log({'critic1_loss': critic_1_loss,
    #            'critic2_loss': critic_2_loss,
    #            'policy_loss': policy_loss,
    #            'entropy_loss': ent_loss,
    #            'alpha': alpha})

    return args.num_train_repeat


def main():
    args = readParser()
    # print(args.use_disc)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    print("Loading everything on " + str(device))
    print("max_epochs",args.max_epochs)
    if args.env == 'Hopper-v2':
        args.num_epoch          =   125
    # elif args.env == 'HalfCheetah-v2':
    #     args.num_epoch = 400
    #     # args.num_train_repeat = 40
    #     # if args.model == 'mdn':x
    #     #     args.num_train_repeat = 20
    # elif args.env == 'Walker2d-v2':
    #     args.num_epoch = 300
    elif args.env == 'Humanoid-v2':
        args.num_epoch          =   300
        args.pred_hidden_size   =   400
        args.automatic_entropy_tuning = True
    else:
        args.num_epoch          =   300
        
    if args.model == 'mlp':
        args.num_networks = 1

    if args.num_elites > args.num_networks:
        args.num_elites = args.num_networks

    # Initial environment
    if args.env == 'Ant-v2':
        from env.ant import AntTruncatedObsEnv
        env = AntTruncatedObsEnv()
    elif args.env == 'Humanoid-v2':
        from env.humanoid import HumanoidTruncatedObsEnv
        env = HumanoidTruncatedObsEnv()
    else:
        env = gym.make(args.env)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    # Intial agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    # if args.env == 'Humanoid-v2':
    #     agent.target_entropy = -2
    #     agent.alpha = 0.05

    # Initial ensemble model    
    state_size = np.prod(env.observation_space.shape)
    action_size = np.prod(env.action_space.shape)
    if args.method == 'vaml':
        if args.env == 'Humanoid-v2':
            pred_hidden_size = (400, 400, 400, 400)#,400, 400, 400, 400)
        else:
            pred_hidden_size = (200, 200, 200, 200)
        env_model = VAML(state_size+action_size, state_size+args.reward_size,
                        hidden_dims=pred_hidden_size)
        env_model.set_gradient_buffer(args,env.observation_space.shape)
        env_model.set_agent(agent)
        env_model.add_mse   =   True
    elif args.method == 'vaml_ensemble':
        if args.env == 'Humanoid-v2':
            pred_hidden_size = (400, 400, 400, 400)#,400, 400, 400, 400)
        else:
            pred_hidden_size = (200, 200, 200, 200)
        env_model = VAML_ensemble(args.num_networks, args.num_elites, state_size,
                                    action_size, args.reward_size, args.pred_hidden_size)
        env_model.set_gradient_buffer(args,env.observation_space.shape)
        env_model.set_agent(agent)
        env_model.add_mse   =   True
    else:
        if args.model == 'ensemble':
            env_model = ProbEnsemble(args.num_networks, args.num_elites, state_size,
                                    action_size, args.reward_size, args.pred_hidden_size)
        elif args.model == 'mlp':
            assert (args.num_networks == 1)
            env_model = ProbEnsemble(args.num_networks, args.num_elites, state_size,
                                    action_size, args.reward_size, args.pred_hidden_size)
        elif args.model == 'mixensemble':
            env_model = MixtureEnsemble(args.num_networks, state_size,
                                    action_size, args.reward_size, args.pred_hidden_size)
        elif args.model == 'ens-mdn':
            if args.env == 'Humanoid-v2':
                pred_hidden_size = (400, 400, 400, 400)
            else:
                pred_hidden_size = (200, 200, 200, 200)
            env_model = EnsembleMDN(args.num_networks, args.num_elites, state_size,
                                    action_size, args.reward_size, hidden_dims=pred_hidden_size)
        elif args.model == 'mdn':
            if args.env == 'Humanoid-v2':
                pred_hidden_size = (400, 400, 400, 400)#,400, 400, 400, 400)
            else:
                pred_hidden_size = (200, 200, 200, 200)
            env_model = MDN(state_size+action_size, state_size+args.reward_size,
                            hidden_dims=pred_hidden_size)
        elif args.model == 'mdn-gaussian':
            if args.env == 'Humanoid-v2':
                pred_hidden_size = (400, 400, 400, 400)
            else:
                pred_hidden_size = (200, 200, 200, 200)
            env_model = MDNGaussian(state_size+action_size, state_size+args.reward_size,
                            hidden_dims=pred_hidden_size)
        elif args.model == 'wider-mdn':
            if args.env == 'Humanoid-v2':
                pred_hidden_size = (1000, 1000, 750, 600)
            else:
                pred_hidden_size = (1000, 1000, 750, 600)
            env_model = MDN(state_size+action_size, state_size+args.reward_size,
                            hidden_dims=pred_hidden_size)
        elif args.model == 'mdn-var':
            if args.env == 'Humanoid-v2':
                pred_hidden_size = (400, 400, 400, 400)
            else:
                pred_hidden_size = (200, 200, 200, 200)
            env_model = MDNVar(state_size+action_size, state_size+args.reward_size,
                            hidden_dims=pred_hidden_size)
        elif args.model == 'dropout':
            from mc_dropout_model import ProbDropOutEnsemble
            env_model = ProbDropOutEnsemble(args.num_networks, args.num_elites, state_size,
                                    action_size, args.reward_size, args.pred_hidden_size)
        else:
            raise NotImplementedError
    env_model.to(device)

    # Predict environments
    predict_env = PredictEnv(env_model, args.env, args.model)

    # Initial pool for env
    env_pool                =   ReplayMemory(args.replay_size, exp_size = args.policy_pool_size)
    cur_pol_pool            =   ReplayMemory(args.policy_pool_size)


    # Initial pool for model
    rollouts_per_epoch      =   args.rollout_batch_size * args.epoch_length / args.model_train_freq
    model_steps_per_epoch   =   int(args.rollout_length * rollouts_per_epoch)
    new_pool_size           =   args.model_retain_epochs * model_steps_per_epoch
    model_pool              =   ReplayMemory(new_pool_size)

    # Sampler of environment
    env_sampler             =   EnvSampler(env)
    debug_sampler           =   EnvSampler(env)

    # Get shape of observation and action for agent
    state_size              =   np.prod(env.observation_space.shape)
    action_size             =   np.prod(env.action_space.shape)
    
    # Initialize TOM occupany matching pipeline - TOM object, Discriminator obj
    tom                     =   SMODICE_TOM(args,state_size, action_size,device)   
    disc                    =   Discriminator_SAS(state_size, action_size, hidden_dim=args.disc_hidden, device=device).to(device)
                     
    train(args, env_sampler, predict_env, agent, env_pool, model_pool, cur_pol_pool, tom, disc,device, debug_sampler)


if __name__ == '__main__':
    main()