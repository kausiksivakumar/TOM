import torch
from mdn import MDN
import numpy as np
import itertools
from tqdm import tqdm
# import wandb

class VAML(MDN):
    def __init__(self, input_dim, output_dim, hidden_dims=..., mixture_size=10, lr=0.001):
        super().__init__(input_dim, output_dim, hidden_dims, mixture_size, lr)
        
        # VAML settings
        self.device         =   torch.device("cuda")
        self.bound_clipping = True
        self.bound_clipping_quantile = 0.95
        self.use_vaml = False
        self.use_scaling = True
        self.mean_g      = False
        self.add_mse     = False
        self.use_all_vf  = True
    
    def set_agent(self, agent):
        # agent -> SAC policy
        self._agent = agent
        self.gradients[:] = 0.0
        self.eval_gradients[:] = 0.0
        self.known_gradients[:] = 0.0
        self.known_eval_gradients[:] = 0.0
    
    def set_gradient_buffer(self, args, obs_shape):
        # obs_shape -> env.observation_space.shape

        # Example for Hopper this would be 125k
        dataset_size    =   args.num_epoch*1000 +5000

        if self.use_all_vf:
            self.gradients = torch.zeros(
                (dataset_size, 4, obs_shape[0]), device=self.device
            )
            self.eval_gradients = torch.zeros(
                (dataset_size, 4, obs_shape[0]), device=self.device
            )
        else:
            self.gradients = torch.zeros(
                (dataset_size, 2, obs_shape[0]), device=self.device
            )
            self.eval_gradients = torch.zeros(
                (dataset_size, 2, obs_shape[0]), device=self.device
            )

        self.known_gradients = torch.zeros(
            (dataset_size, 1), dtype=torch.bool, device=self.device
        )
        self.known_eval_gradients = torch.zeros(
            (dataset_size, 1), dtype=torch.bool, device=self.device
        )

    def train(self, inputs, targets, next_state, batch_size=256, holdout_ratio=0.2,weights = None,
              max_logging=5000, max_epochs_since_update=5, max_epochs=50,
              device='cuda:0'):
        self.batch_size = batch_size
        # TODO:Get weights for TOM sampling
        if(weights is None):
            weights                 =   np.ones((inputs.shape[0],1))
        #Clipping weights to 10
        weights                     =   weights.clip(None,10)
        self._max_epochs_since_update = max_epochs_since_update
        self._snapshot = (None, 1e10) # keeping track of the best val epoch
        self._epochs_since_update = 0 

        def shuffle_rows(arr1,arr2,arr3, inp_idx):
            # idxs = np.argsort(np.random.uniform(size=arr1.shape[0]), axis=-1)
            idxs = np.random.permutation(arr1.shape[0])
            return arr1[idxs], arr2[idxs], arr3[idxs], inp_idx[idxs]

        # num_holdout = int(inputs.shape[0] * holdout_ratio)
        num_holdout = min(int(inputs.shape[0] * holdout_ratio), max_logging)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, holdout_inputs = inputs[permutation[num_holdout:]], inputs[permutation[:num_holdout]][:10000]
        targets, holdout_targets = targets[permutation[num_holdout:]], targets[permutation[:num_holdout]][:10000]
        next_obs, holdout_next_obs  =   next_state[permutation[num_holdout:]], next_state[permutation[:num_holdout]][:10000]   
        
        idx_inp,idx_val =   permutation[num_holdout:], permutation[:num_holdout][:10000] 
        
        # normalization using the training set
        self.fit_input_stats(inputs, targets)
        
        input_val   = torch.from_numpy(holdout_inputs).float().to(device)
        target_val  = torch.from_numpy(holdout_targets).float().to(device)
        input_val   = (input_val - self.inputs_mu) / self.inputs_sigma
        target_val  = (target_val - self.targets_mu) / self.targets_sigma
        next_obs_val= torch.from_numpy(holdout_next_obs).float().to(device)

        if max_epochs is not None:
            epoch_iter = range(max_epochs)
        else:
            epoch_iter = itertools.count()

        #Evaluate dataset here to make critic requires grad as True
        # best_val_score                  =   self.eval_batch(input_val,target_val,idx_val,next_obs_val)

        for epoch in tqdm(epoch_iter):
            # idxs                        =   np.random.choice(inputs.shape[0], size=[100*batch_size], p= train_prob)
            idxs                        =   np.arange(len(inputs))
            batch_loss                  =   0.0
            # for idx,batch_num in enumerate(range(0,100*batch_size,batch_size)):
            for i,batch_num in enumerate(range(min(100,int(np.floor(idxs.shape[-1] / self.batch_size))))):
                batch_idxs = idxs[batch_num * batch_size:(batch_num + 1) * batch_size]
                # batch_idxs  =   idxs[batch_num:batch_num+batch_size]
                input       =   torch.from_numpy(inputs[batch_idxs]).float().to(device)
                target      =   torch.from_numpy(targets[batch_idxs]).float().to(device)
                next_ob     =   torch.from_numpy(next_obs[batch_idxs]).float().to(device)
                idx         =   torch.from_numpy(idx_inp[batch_idxs]).to(device)
                if self.fit_input:
                    input = (input - self.inputs_mu) / self.inputs_sigma
                    target = (target - self.targets_mu) / self.targets_sigma

                # pred_distribution = self(input)
                
                #VAML loss here
                loss = self.loss(input, target,idx,next_ob)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_loss  +=  loss.item()
                
            batch_loss/=i
            inputs,targets,next_obs,idx_inp =   shuffle_rows(inputs,targets,next_obs,idx_inp)

            # idxs = shuffle_rows(idxs)
            # Since val might contain 0 elements
            if(len(input_val)==0):
                rmse    =   input[0][0]
                val_loss=   input[0][0]
                continue
            # input_val,target_val,next_obs_val,idx_val =   shuffle_rows(input_val,target_val,next_obs_val,idx_val)
            # input_val                                 =   input_val[:1000]
            # target_val                                =   target_val[:1000]
            # next_obs_val                              =   next_obs_val[:1000]
            # idx_val                                   =   idx_val[:1000]

            val_loss    =   self.eval_batch(input_val,target_val,idx_val,next_obs_val)
            break_train =   self._save_best(epoch, val_loss)
            
            if break_train:
                break
            # print(f"Epoch {epoch} Train {loss.cpu().detach().item():.3f} Val {val_loss.cpu().detach().item():.3f}")
            # wandb.log({'train_loss':batch_loss,'val_loss':val_loss})
        # Just to keep consistent with ensemble API
        self.elite_model_idxes = [0]
        return batch_loss, val_loss
    
    def eval_batch(self,input,target,idx_val,next_obs):
        eval_loss   =   0.0
        tot_idxs    =   np.arange(idx_val.shape[-1])
        for i,batch_num in enumerate(range(int(np.floor(idx_val.shape[-1] / self.batch_size)))):
            # batch_idxs = idx_val[batch_num * self.batch_size:(batch_num + 1) * self.batch_size]
            batch_idxs = tot_idxs[batch_num * self.batch_size:(batch_num + 1) * self.batch_size]
            inp     =   input[batch_idxs]#.to('cuda:0')   
            tar     =   target[batch_idxs]#.to('cuda:0')
            # inp     =   (inp - self.inputs_mu) / self.inputs_sigma
            # tar     =   (tar - self.targets_mu) / self.targets_sigma
            idx     =   idx_val[batch_idxs]
            next_ob =   next_obs[batch_idxs]#.to('cuda:0')
            loss    =   self._vaml_loss(inp,tar,idx,next_ob,eval=True)
            # loss    =   self._vaml_loss(input,target,idx_val,next_obs,eval=True)
            if(self.add_mse):
                output  =   self.forward(inp)
                loss    +=  -output.log_prob(tar).mean(-1,keepdim=True)
            eval_loss   +=  loss.mean().detach()
        return eval_loss/i
        # return loss.mean().detach()
    
    def loss(self,input,target,idx_val,next_obs):
        loss    =   self._vaml_loss(input,target,idx_val,next_obs,eval=False)
        if(self.add_mse):
            output  =   self.forward(input)
            loss    +=  -output.log_prob(target).mean(-1,keepdim=True)
        return loss.mean()

    def values(self,next_obs):
        # This is directly taken from VaGram repository 
        self._agent.policy.requires_grad        =   False
        self._agent.critic.requires_grad        =   False
        self._agent.critic_target.requires_grad =   False
        _,_,next_action                         =   self._agent.policy.sample(next_obs)   
        values                                  =   self._agent.critic(next_obs,\
                                                                next_action.detach())
        values_target                           =   self._agent.critic_target(next_obs,\
                                                                next_action.detach())
        if self.use_all_vf:
            all_values  = torch.stack([*values, *values_target], 0)
        else:
            all_values  = torch.stack([*values_target], 0)
        return all_values.squeeze(1)


    def _vaml_loss(self,model_in,target,idx,next_obs,eval=False):
        # This function is mostly taken as is from VaGram repo
        output                                  =   self.forward(model_in)
        
        next_obs.requires_grad                  =   True
        # These two do not do anything, idk why they are here (can't freeze model this way)
        self._agent.critic.requires_grad        =   False
        self._agent.critic_target.requires_grad =   False
        '''
        This is how model should be frozen
        # Set requires grad attribute as False for critic and target 
        for param in self._agent.critic.parameters():
            param.requires_grad = False
        for param in self._agent.critic_target.parameters():
            param.requires_grad = False
        '''
        vf_pred                                 =   self.values(next_obs)
        vaml_loss                               =   0.0

        for i, vf in enumerate(vf_pred):
            if eval and torch.all(self.known_eval_gradients[idx]):
                # Is the unsqueeze needed here?
                g   =   self.eval_gradients[idx,i]
            elif torch.all(self.known_gradients[idx]):
                g   =   self.gradients[idx, i]
            else:
                if i == len(vf_pred)    -   1:
                    vf.sum().backward(retain_graph=False)
                else:
                    vf.sum().backward(retain_graph=True)
                g   =   next_obs.grad.clone().detach().squeeze()
                if eval:
                    self.eval_gradients[idx, i] = g
                else:
                    self.gradients[idx, i]      = g

            if self.bound_clipping:
                norms           = torch.sqrt(torch.sum(g**2, -1))
                quantile_bound  = np.quantile( \
                                        norms.detach().cpu().numpy(), \
                                        self.bound_clipping_quantile \
                                        )
                norms           =   norms.unsqueeze(-1)
                g               = torch.where(
                                        torch.logical_or(norms < quantile_bound\
                                            , norms < 100000), g, (quantile_bound / norms)\
                                                 * g\
                                            ).detach()
            else:
                g               =   g.clone().detach()
            

            '''
            # Kausik alter -- reward weight is mean g 
            g_mean              =   g.mean(axis=-1).view(-1,1)
            g                   =   torch.hstack((g,g_mean))
            if self.use_vaml:
                vaml_loss       +=  (
                                    torch.sum(
                                    g * -output.log_prob(target), -1,\
                                         keepdim=True
                                    )
                                    ** 2
                                    )
            elif self.use_scaling:
                vaml_loss       +=  g * -output.log_prob(target)
            '''
            # Making all grads positive -- since I am just weighting it
            if self.use_vaml:
                vaml_loss       +=  (
                                    torch.sum(
                                    g * -output.log_prob(target)[:,:-1], -1,\
                                         keepdim=True
                                    )
                                    ** 2
                                    )
            elif self.use_scaling:
                vaml_loss       +=  g * -output.log_prob(target)[:,:-1]
            
            elif self.mean_g:
                g_mean          =   g.mean(axis=-1,keepdim=True)
                vaml_loss       +=  -g_mean*output.log_prob(target).mean(axis=-1,keepdim=True)
            
            self._agent.critic.zero_grad()
            self._agent.critic_target.zero_grad()
            if target.grad is not None:
                target.grad[:] = 0.0
        
        if eval:
            self.known_eval_gradients[idx] = True
        else:
            self.known_gradients[idx]      = True

        next_obs.requires_grad = False
        self._agent.critic.requires_grad = True
        self._agent.critic_target.requires_grad = True
        vaml_loss /= len(vf_pred)
        
        # For reward
        # wandb.log({'vaml_comp':vaml_loss.mean(),'reward_comp':-output.log_prob(target)[:,-1:].mean()})
        if(not self.mean_g):
            vaml_loss               +=  -output.log_prob(target)[:,-1:]
        return vaml_loss