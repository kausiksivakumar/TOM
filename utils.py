from socket import AF_X25
# import d4rl
import gym
import numpy as np
import torch
import gc
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
from PIL import Image
import torch

def load_d4rl_dataset(env_name='halfcheetah-expert-v0'):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)
    # dataset['rewards'] = np.expand_dims(dataset['rewards'], axis=1)
    # dataset['terminals'] = np.expand_dims(dataset['terminals'], axis=1)
    return env, dataset


def load_normalized_dataset(env_name='hopper', dataset_name='medium-replay-v0'):
    x_train, y_train, x_test, y_test = np.load(f'data/{env_name}-{dataset_name}-normalized-data.npy', allow_pickle=True)
    return x_train,y_train,x_test, y_test


def multistep_dataset(env, h=2, terminate_on_end=False, **kwargs):
    dataset = env.get_dataset(**kwargs)
    N = dataset['rewards'].shape[0]

    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N - h):
        skip = False
        for j in range(i, i + h - 1):
            if bool(dataset['terminals'][j]) or dataset['timeouts'][j]:
                skip = True
        if skip:
            continue

        obs = dataset['observations'][i]
        new_obs = dataset['observations'][i + h]
        action = dataset['actions'][i:i + h].flatten()
        reward = dataset['rewards'][i + h - 1]
        done_bool = bool(dataset['terminals'][i + h - 1])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i + h - 1]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }


def format_samples_for_training(samples):
    obs = samples['observations']
    act = samples['actions']
    next_obs = samples['next_observations']
    rew = samples['rewards']
    delta_obs = next_obs - obs
    inputs = np.concatenate((obs, act), axis=-1)
    outputs = np.concatenate((rew.reshape(rew.shape[0], -1), delta_obs), axis=-1)

    # inputs = torch.from_numpy(inputs).float()
    # outputs = torch.from_numpy(outputs).float()

    return inputs, outputs


def create_data_loader(replay_buffer, expert_buffer, batch_size=64):
    '''
    Creates a dataloader for discriminator
    '''
    if(len(expert_buffer)   < 1000):
        raise Exception("Current policy pool size is just 1000")

    env_samples         =   replay_buffer.sample(len(replay_buffer))
    env_dataset         =   TensorDataset(torch.FloatTensor(np.hstack((env_samples[0],env_samples[1],env_samples[3])).astype(np.float32)))
    env_loader          =   DataLoader(env_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    exp_samples         =   expert_buffer.sample(len(expert_buffer))
    exp_dataset         =   TensorDataset(torch.FloatTensor(np.hstack((exp_samples[0],exp_samples[1],exp_samples[3])).astype(np.float32)))
    exp_loader          =   DataLoader(exp_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # return {'env_samples':env_samples, 'env_loader': env_loader, 'cur_samples': exp_samples, 'cur_loader':exp_loader}
    return (env_samples, env_loader, exp_samples, exp_loader)

def permute_and_pass(samples, size = None):
    '''
    Converts tuples of (s,a,s',r,term) to separate arrays ->  samples 'size' values -> converts to tensor and returns them
    '''
    state,action,reward,next_state,terminal,disc_reward,_     =   samples 
    
    if(size is not None):
        idx                                     =   np.random.randint(len(state), size = size)
    
    else:
        idx                                     =   np.arange(state.shape[0])
    
    state                                       =   torch.from_numpy(state[idx]).float()       
    action                                      =   torch.from_numpy(action[idx]).float() 
    reward                                      =   torch.from_numpy(reward[idx]).float() 
    next_state                                  =   torch.from_numpy(next_state[idx]).float() 
    terminal                                    =   torch.from_numpy(terminal[idx]).float() 
    disc_reward                                 =   torch.from_numpy(disc_reward[idx]).float() 
    return (state,action,reward,next_state,terminal,disc_reward)

def bin_rewards_cur_pol(sas_inputs,sas_curpol):
    '''
    Inputs - S,A,S' concatenated
    Output - shape of nx1, where otp is 1 if sas_inputs in cur_policy_pool
    '''
    return torch.sum(~torch.cdist(sas_inputs, sas_curpol, 1).bool(),dim=1).bool().view(-1,1)*1
    
def sample_init_state(replay_buffer,batch_size):
    '''
    Samples initial state from env_pool.init_indices. Samples batch_size amount of initial states for Q learning TOM
    Args: env_buffer - obj of ReplayMemory class
          batch_size - amount of init_state samples needed
    '''
    init_state                                          =   np.array(replay_buffer.init_state)
    if(len(replay_buffer.init_state)<batch_size):
        init_state                                      =   init_state
    else:
        init_idxs                                       =   np.random.randint(len(init_state),size = batch_size)
        init_state                                      =   init_state[init_idxs]
    init_state                                          =   torch.from_numpy(init_state).float()
    return init_state

def cal_weights(state,action,reward,next_state,terminal,disc_reward,discriminator,tom,policy,use_disc=False):
    '''
    A helper function to get the weights given sars'td
    '''
    with torch.no_grad():
        th_state                                =   torch.from_numpy(state).float()
        th_act                                  =   torch.from_numpy(action).float()
        th_next_state                           =   torch.from_numpy(next_state).float()
        th_done                                 =   torch.from_numpy(terminal).float()
        disc_reward                             =   torch.from_numpy(disc_reward).float()

        if(use_disc):
            disc_input                          =   torch.cat([th_state, th_act,th_next_state], axis=1).float()
            disc_reward                         =   discriminator.predict_reward(disc_input)
        w_e                                     =   tom.get_weight(th_state, th_act, disc_reward, reward, th_next_state, th_done, policy=policy)
        del th_state,th_act,th_next_state,th_done,disc_reward
        gc.collect()
    return w_e[0]

def save_policy_gif(args, policy, env_sampler, epoch_step):
    '''
    A function to save the final policy state_dict and its associated gifs
    (i.e) rendering headless
    '''
    gif_name                                                =   str(args.env)+'_'+str(args.method)+str(args.seed)+'_e'+str(epoch_step)+'.gif'
    policy_name                                             =   str(args.env)+'_'+str(args.method)+str(args.seed)+'.pth'

    policy_path                                             =   "./out/policy/"
    gif_path                                                =   "./out/gifs/"
    isExistgif                                              =   os.path.exists(gif_path)
    isExistpolicy                                           =   os.path.exists(policy_path)
    
    # Create directories if they don't exist
    if not isExistgif:
        os.makedirs(gif_path)
    if not isExistpolicy:
        os.makedirs(policy_path)

    # save policy
    torch.save(policy.policy.state_dict(), policy_path+policy_name)

    # save gif
    env_sampler.current_state           =   None
    img_list                            =   []
    # Since epoch length = 1000
    for t in range(args.epoch_length):
        img                             =   env_sampler.env.render(mode='rgb_array')
        img_list.append(Image.fromarray(img))
        # Take step in env on policy
        _, _, _, _, done, _             =   env_sampler.sample(policy, eval_t=True)
        
        if done:
            break
    print("Ran for steps", t)
    # Saving gif
    img_list[0].save(gif_path+gif_name, save_all=True, append_images=img_list[1:], duration=50, loop=0)


def weight_vs_position(env_pool, disc, tom, policy, use_disc = False, buffer_size  =   3000):
    cur_idxs            =   None
    old_idxs            =   None
    if(len(env_pool)    >=  env_pool.capacity):
        # Getting the current position in env pool - current policy's population
        cur_pos         =   env_pool.position - 1
        start_pos       =   cur_pos - buffer_size
        cur_idxs        =   np.arange(start_pos,cur_pos)
        cur_idxs[np.where(cur_idxs<0)]+=len(env_pool)

        # Get the current position of oldest policy
        cur_pos             =   env_pool.position
        end_pos             =   cur_pos   +   buffer_size
        old_idxs            =   np.arange(cur_pos,end_pos)
        old_idxs[np.where(old_idxs>=len(env_pool))]-=len(env_pool)

    else:
        cur_pos         =   env_pool.position - 1
        start_pos       =   cur_pos - buffer_size
        cur_idxs        =   np.arange(start_pos,cur_pos)
        cur_idxs[np.where(cur_idxs<0)]+=len(env_pool)

        # Get the current position of oldest policy
        cur_pos             =   0
        end_pos             =   cur_pos   +   buffer_size
        old_idxs            =   np.arange(cur_pos,end_pos)
        old_idxs[np.where(old_idxs>=len(env_pool))]-=len(env_pool)
    
    cur_state,cur_action,cur_reward,cur_next_state,cur_terminal,cur_disc_reward =   idx_buffer(cur_idxs,env_pool)
    old_state,old_action,old_reward,old_next_state,old_terminal,old_disc_reward =   idx_buffer(old_idxs,env_pool)
    old_weight                                                                  =   cal_weights(old_state,old_action,old_reward,old_next_state,old_terminal,old_disc_reward,disc,tom,policy,use_disc=use_disc)
    cur_weight                                                                  =   cal_weights(cur_state,cur_action,cur_reward,cur_next_state,cur_terminal,cur_disc_reward,disc,tom,policy,use_disc=use_disc)

    avg_vals    =   {"old_weight":old_weight.mean(),
                     "cur_weight":cur_weight.mean()}
    
    return avg_vals

def idx_buffer(idxs,env_pool):
    state                =   np.array([env_pool.buffer[idx][0] for idx in idxs])
    action               =   np.array([env_pool.buffer[idx][1] for idx in idxs])
    reward               =   np.array([env_pool.buffer[idx][2] for idx in idxs])
    next_state           =   np.array([env_pool.buffer[idx][3] for idx in idxs])
    terminal             =   np.array([env_pool.buffer[idx][4] for idx in idxs])
    disc_reward          =   env_pool.disc_rewards[idxs]


    return (state,action,reward,next_state,terminal,disc_reward)

def model_pred_error(debug_sampler,pred_env,agent,epoch_length=5):
    '''
    Calculates epoch_length step model error and returns it. States start from env.reset()
    '''
    state                       =   debug_sampler.env.reset()
    debug_sampler.current_state =   state
    error                       =   []
    action_buffer               =   []
    next_state_buffer           =   []
    tot_error   =   0
    for t in range(epoch_length):
        cur_state, action, next_state, reward, done, info = debug_sampler.sample(agent, eval_t=True)
        action_buffer.append(action)
        next_state_buffer.append(next_state)
        if(done):
            next_state_flag     =   next_state
            break
    
    if(len(next_state_buffer)   !=  epoch_length):
        next_state_extend           =   [next_state_flag]*(epoch_length -   len(next_state_buffer))
        next_state_buffer.extend(next_state_extend)

    pred_next_state             =   []
    for t in range(len(action_buffer)):
        state, rewards, terminals, info   =   pred_env.step(state, action_buffer[t])
        pred_next_state.append(state)

    if(len(pred_next_state)   !=  epoch_length):
        next_state_extend           =   [state]*(epoch_length -   len(pred_next_state))
        pred_next_state.extend(next_state_extend)
    
    pred_next_state             =   np.array(pred_next_state)
    true_next_state             =   np.array(next_state_buffer)
    
    return(np.linalg.norm(pred_next_state - true_next_state,axis=1))

def update_litm_prob(env_pool, policy_train_steps, litm_decay, prev_length):
    '''
    Updates the probability weights of LITM like mentioned in the paper
    '''
    # prob                    =   env_pool.prob
    if(policy_train_steps   ==  0):
        # env_pool.num_curpol             +=   1
        temp                            =   (1  -   litm_decay)/env_pool.num_curpol
        if(prev_length      >=  env_pool.capacity):
            c = env_pool.prob[env_pool.position-1]/(len(env_pool) - env_pool.num_curpol)
            env_pool.prob   +=  c
            env_pool.prob[max(0,env_pool.position - env_pool.num_curpol):env_pool.position]    =   temp   
            if((env_pool.position - env_pool.num_curpol)<0):
                env_pool.prob[env_pool.position - env_pool.num_curpol:]    =   temp   

        else:
            env_pool.prob[env_pool.position - env_pool.num_curpol:]    =   temp
            env_pool.prob                =   np.concatenate((env_pool.prob,np.array([temp])))  
    
    
    else:
        env_pool.num_curpol             =   1
        if(prev_length>=env_pool.capacity):
            env_pool.prob                        =   (env_pool.prob)*litm_decay/(sum(env_pool.prob) - env_pool.prob[env_pool.position-1])
            env_pool.prob[env_pool.position - 1] =   1 - litm_decay
        else:
            env_pool.prob               =   env_pool.prob*litm_decay
            new_s_prob                  =   sum(env_pool.prob)*(1 - litm_decay)/litm_decay
            env_pool.prob               =   np.concatenate((env_pool.prob,np.array([new_s_prob])))
    
    # print("prob_sum",env_pool.prob.sum())
    env_pool.prob   =   env_pool.prob/(env_pool.prob.sum())
    # print("prob_sum",env_pool.prob.sum())
    
def get_weights_pos_bar(args,env_pool,tom,discriminator,policy,epoch=0):
    '''
    This returns a dictionary with transition weights in bins -- from old to new
    '''
    weights_dict        =   {}
    f                   =   plt.figure(figsize=(10,5))
    # TODO: Get the indexes of samples from oldest to newest
    idxes               =   None
    weights             =   None
    if(len(env_pool)    >=  env_pool.capacity):
        # Get the current position of newest policy
        center      =   env_pool.position - 1
        idxes       =   np.concatenate((np.arange(center,env_pool.capacity),np.arange(0,center)))
    else:
        idxes       =   np.arange(0,len(env_pool))

    # TODO: Get weights using TOM for all samples. Save with matplotlib
    if("tom" in args.method):
        state,action,reward,next_state,terminal,disc_reward =   idx_buffer(idxes,env_pool)
        weights                                             =   cal_weights(state,action,reward,next_state,terminal,disc_reward,discriminator,tom,policy,use_disc=args.use_disc).flatten()
        weights                                             =   (weights/weights.sum())
    elif("litm" in args.method):
        weights                                             =   env_pool.prob[idxes]
    else:
        weights                                             =   np.array([1]*len(env_pool))/len(env_pool)

    # TODO: Plot and save
    weights                                                 =   weights.reshape((100,-1)).mean(axis=-1)

    f                                                       =   plt.figure(figsize=(7,3))
    ax                                                      =   f.add_subplot(111)
    ax.bar(np.arange(2*len(weights))[::2],weights,alpha=0.3)
    ax.plot(np.arange(2*len(weights))[::2],weights,color='black')
    save_path                                               =   "./plots/"+str(args.method)+'/'+str(args.env)+'/'
    isExist                                                 =   os.path.exists(save_path)
    if not isExist:
        os.makedirs(save_path)
    plt.title(str(args.method)+"_weights_ep"+str(0))
    plt.xticks([])
    plt.xlabel("Temporal progression of samples collected")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    # plt.ylabel("Average weights (100 bins total)")    
    plt.savefig(save_path+'_e_'+str(epoch)+'_s_'+str(args.seed)+'.png')

def shuffle_rows(arr):
    idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
    return arr[np.arange(arr.shape[0])[:, None], idxs]


def batch_generator(index_array, batch_size):
    index_array = shuffle_rows(index_array)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= index_array.shape[1]:
            batch_count = 0
            index_array = shuffle_rows(index_array)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield index_array[:, start:end]

def model_error(debug_sampler, agent, predict_env):
    # Get current state
    model_err       =   0.0
    for i in range(5):
        cur_state, action, next_state, _, _, _   =   debug_sampler.sample(agent,eval_t=True)
        pred_next_state, _, _,_                  =   predict_env.step(cur_state, action)
        model_err                                +=   np.sqrt(sum((next_state - pred_next_state.squeeze())**2))
    
    model_err/=5
    return model_err

def truncated_linear(
    min_x: float, max_x: float, min_y: float, max_y: float, x: float
) -> float:
    """Truncated linear function.

    Implements the following function:
        f1(x) = min_y + (x - min_x) / (max_x - min_x) * (max_y - min_y)
        f(x) = min(max_y, max(min_y, f1(x)))

    If max_x - min_x < 1e-10, then it behaves as the constant f(x) = max_y
    """
    if max_x - min_x < 1e-10:
        return max_y
    if x <= min_x:
        y: float = min_y
    else:
        dx = (x - min_x) / (max_x - min_x)
        dx = min(dx, 1.0)
        y = dx * (max_y - min_y) + min_y
    return y