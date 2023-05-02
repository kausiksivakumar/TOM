import numpy as np 
import torch 
from networks_pytorch import TanhMixtureNormalPolicy, TanhNormalPolicy, ValueNetwork, QNetwork, weights_init_
import pdb
import gc

class SMODICE_TOM(object):
    def __init__(self, args,state_spec, action_spec,device):

        self._gamma         =   args.gamma
        self._hidden_sizes  =   args.hidden_sizes
        self._batch_size    =   args.m_batch_size
        self._f             =   args.f
        self._lr            =   3e-4
        self._max_epochs    =   args.max_epochs
        self._q_l2_reg      =   args.q_l2_reg
        self.method         =   args.method 
        self.device         =   device

        self._iteration     =   0
        self._optimizers    =   dict()
 
        # self._v_network = ValueNetwork(observation_spec, hidden_sizes=self._hidden_sizes).to(self.device)
        self._q_network     =   QNetwork(state_spec,action_spec,hidden_sizes=self._hidden_sizes).to(self.device)

        self._optimizers['q'] = torch.optim.Adam(self._q_network.parameters(), self._lr, weight_decay=self._q_l2_reg)

        #Creating a delayed target, trying the delayed Q network trick like DQN
        # self._delayed_target  =   QNetwork(state_spec,action_spec,hidden_sizes=self._hidden_sizes).to(self.device)
        self.alpha            =   0.05 #Exponential averaging of target

        # f-divergence functions
        if self._f == 'chi':
            self._f_fn = lambda x: 0.5 * (x - 1) ** 2
            self._f_star_prime = lambda x: torch.relu(x + 1)
            self._f_star = lambda x: 0.5 * x ** 2 + x 
        elif self._f == 'kl':
            self._f_fn = lambda x: x * torch.log(x + 1e-10)
            self._f_star_prime = lambda x: torch.exp(x - 1)
        else:
            raise NotImplementedError()


    def q_loss(self, initial_v_values, e_v, result={}):
        # Compute v loss
        q_loss0 = (1 - self._gamma) * initial_v_values

        if self._f == 'kl':
            q_loss1 = torch.log(torch.mean(torch.exp(e_v))) #This guy sometimes become inf 
        else:
            q_loss1 = torch.mean(self._f_star(e_v))
        
        # print("Iteration: {},val_in_q_loss1_log:{}".format(self._iteration,torch.mean(torch.exp(e_v))))

        q_loss = q_loss0 + q_loss1          #Adding torch.mean to Q_loss 0
        q_loss = torch.mean(q_loss)

        result.update({
            'q_loss0': torch.mean(q_loss0),
            'q_loss1': torch.mean(q_loss1),
            'q_loss': q_loss,
        })

        return result

    def model_learning(self,state,action,next_state,reward,w_e,max_epochs=1,pred_env=None):
        '''
        Directly taken from train_predict_model function in mbpo.py
        '''
        if(pred_env is None):
            raise Exception("You haven't provided a dynamics model for model_learning")

        state                       =   state.cpu().numpy()
        action                      =   action.cpu().numpy()
        reward                      =   reward.cpu().numpy()
        w_e                         =   w_e#.cpu().numpy()
        next_state                  =   next_state.cpu().numpy()
        
        #Convert it to correct inputs for EnsembleDynamicsModel
        delta_state                 =   next_state - state
        inputs                      =   np.concatenate((state, action), axis=-1)
        labels                      =   np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1)

        '''
        To debug
        '''
        # inputs = np.array([[1]*23,[2]*23,[3]*23,[4]*23,[5]*23])
        # labels = labels[:5]
        # w_e = np.array([[1],[2],[3],[4],[5]])
        #TODO: Complete model training with weights given
        val_mse,val_nll             =   pred_env.model.train(inputs, labels, batch_size=self._batch_size, holdout_ratio=0.1,weights=w_e,max_epochs=max_epochs)
        
        return val_mse,val_nll


    def train_step(self, initial_state, state, action, disc_reward,reward, next_state, terminal, pred_env=None, policy = None):
        if(pred_env is None or policy is None):
            raise Exception("You either haven't provided the dynamics model or the policy for TOM Q-learning step")

        # if(self._iteration%1000==0):
        #     self._delayed_target.load_state_dict(self._q_network.state_dict())

        initial_state       = initial_state.to(self.device)
        state               = state.to(self.device)
        action              = action.to(self.device)
        disc_reward         = disc_reward.to(self.device)
        next_state          = next_state.to(self.device)
        terminal            = terminal.unsqueeze(1).to(self.device)

        # Shared network values
        #Get initial v value from Q network (Take n action randomly and average it)
        act_random          =   torch.tensor(policy.select_action(initial_state.cpu())).to(self.device)
        initial_v_values    =   self._q_network([initial_state,act_random])[0]

        #Q value of current state,action
        q_values             =   self._q_network([state,action])[0]

        #Get next v values
        act_random           =  torch.tensor(policy.select_action(next_state.cpu())).to(self.device)
        next_v_values        =  self._q_network([next_state,act_random])[0]

        e_v                  =  disc_reward + (1 - terminal)* self._gamma * next_v_values - q_values


        # compute value function loss (Equation 20 in the paper)
        loss_result          =  self.q_loss(initial_v_values, e_v, result={})

        self._optimizers['q'].zero_grad()
        loss_result['q_loss'].backward()
        self._optimizers['q'].step()
        self._iteration     +=  1

        return loss_result

    def step(self, observation):
        """
        observation: batch_size x obs_dim
        """
        observation = torch.from_numpy(observation).to(self.device)
        action = self._policy_network.deterministic_action(observation)

        return action.detach().cpu(), None


    def train_model_step(self, state, action, disc_reward, reward, next_state, terminal,pred_env=None, policy=None):
        if(pred_env is None or policy is None):
            raise Exception("You either haven't provided the dynamics model or the policy for TOM model-learning step")
        
        w_e,_           =   self.get_weight(state, action, disc_reward, reward, next_state, terminal,policy=policy)
        val_mse,val_nll =   self.model_learning(state,action,next_state,reward,w_e,max_epochs=self._max_epochs, pred_env=pred_env)
        del state,action,disc_reward,next_state,terminal
        return w_e

    def get_weight(self, state, action, disc_reward, reward, next_state, terminal,policy=None):
        if(policy is None):
            raise Exception("You haven't provided policy for get_weight func smodice_pytorch.py")

        with torch.no_grad():

            state               = state.to(self.device)
            action              = action.to(self.device)
            disc_reward         = disc_reward.to(self.device)
            next_state          = next_state.to(self.device)
            terminal            = terminal.unsqueeze(1).to(self.device)
            q_values            = self._q_network([state,action])[0]

            act_random          =  torch.tensor(policy.select_action(next_state.cpu())).to(self.device) 
            next_v_values       = self._q_network([next_state,act_random])[0]

            e_v                 = disc_reward + (1 - terminal)* self._gamma * next_v_values - q_values
            
            # extracting importance weight (Equation 21 in the paper)
            if self._f == 'kl':
                w_e = torch.exp(e_v).detach()
            else:
                w_e = self._f_star_prime(e_v).detach()
            del state,action,disc_reward,next_state,terminal,q_values,act_random,next_v_values
            gc.collect()
            return w_e.cpu().numpy(),e_v.cpu().numpy()
        
    def save_Q(self,path):
        Q_path = path+'-Q.pt'
        torch.save(self._q_network.state_dict(), Q_path)


