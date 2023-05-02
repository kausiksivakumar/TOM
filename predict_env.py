import numpy as np
import torch

# import wandb

class PredictEnv:
    def __init__(self, model, env_name, model_type='ensemble', discriminator=None, rnd_model=None):
        self.model = model
        self.env_name = env_name
        self.model_type = model_type
        self.discriminator = discriminator
        self.rnd_model = rnd_model

    def _termination_fn(self, env_name, obs, act, next_obs):
        prefix = env_name.split('-')[0]
        if env_name == "Hopper-v2" or prefix == 'hopper':
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done =  np.isfinite(next_obs).all(axis=-1) \
                        * np.abs(next_obs[:,1:] < 100).all(axis=-1) \
                        * (height > .7) \
                        * (np.abs(angle) < .2)

            done = ~not_done
            done = done[:,None]
            return done

        elif env_name == 'HalfCheetah-v2' or prefix == 'halfcheetah':
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            done = np.array([False]).repeat(len(obs))
            done = done[:, None]
            return done

        elif env_name == "Walker2d-v2" or prefix == 'walker2d':
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done =  (height > 0.8) \
                        * (height < 2.0) \
                        * (angle > -1.0) \
                        * (angle < 1.0)
            done = ~not_done
            done = done[:,None]
            return done
        elif env_name == "Ant-v2" or prefix == 'ant':
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            x = next_obs[:, 0]
            not_done = np.isfinite(next_obs).all(axis=-1) \
                       * (x >= 0.2) \
                       * (x <= 1.0)

            done = ~not_done
            done = done[:, None]
            return done
        elif env_name == "Humanoid-v2" or prefix == 'humanoid':
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            z = next_obs[:, 0]
            done = (z < 1.0) + (z > 2.0)

            done = done[:, None]
            return done
    def _get_logprob(self, x, means, variances):

        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        log_prob = -1/2 * (k * np.log(2*np.pi) + np.log(variances).sum(-1) + (np.power(x-means, 2)/variances).sum(-1))

        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means,0).mean(-1)

        return log_prob, stds

    def step(self, obs, act, deterministic=False):
        if(len(obs.shape) == 1):
            obs =   obs[None]
            act =   act[None]
        assert len(obs.shape) == 2

        inputs = np.concatenate((obs, act), axis=-1)
        if self.model_type == 'ensemble' or self.model_type == 'dropout' or self.model_type == 'mlp' or self.model_type == 'mdn-gaussian':
            ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)
            ensemble_model_means[:, :, 1:] += obs
            ensemble_model_stds = np.sqrt(ensemble_model_vars)
            if deterministic:
                ensemble_samples = ensemble_model_means
            else:
                ensemble_samples = ensemble_model_means + np.random.normal(
                    size=ensemble_model_means.shape) * ensemble_model_stds

            num_models, batch_size, _ = ensemble_model_means.shape
            model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)

        elif self.model_type == 'mdn' or self.model_type == 'ens-mdn' or \
                self.model_type == 'wider-mdn' or self.model_type == 'mdn-var' or self.model_type == 'mixensemble':
            ensemble_samples = self.model.predict(inputs)
            ensemble_samples[:,:, 1:] += obs

            _, batch_size, _ = ensemble_samples.shape
            model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
        else:
            raise NotImplementedError

        batch_idxes = np.arange(0, batch_size)
        samples = ensemble_samples[model_idxes, batch_idxes]

        # model_means = ensemble_model_means[model_idxes, batch_idxes]
        # model_stds = ensemble_model_stds[model_idxes, batch_idxes]
        # log_prob, dev = self._get_logprob(samples, ensemble_model_means, ensemble_model_vars)

        rewards, next_obs = samples[:,:1], samples[:,1:]
        terminals = self._termination_fn(self.env_name, obs, act, next_obs)

        # batch_size = model_means.shape[0]
        # return_means = np.concatenate((model_means[:,:1], terminals, model_means[:,1:]), axis=-1)
        # return_stds = np.concatenate((model_stds[:,:1], np.zeros((batch_size,1)), model_stds[:,1:]), axis=-1)
        # if return_single:
        #     next_obs = next_obs[0]
        #     return_means = return_means[0]
        #     return_stds = return_stds[0]
        #     rewards = rewards[0]
        #     terminals = terminals[0]
        #
        # info = {'mean': return_means, 'std': return_stds, 'log_prob': log_prob, 'dev': dev}

        info = {}
        return next_obs, rewards, terminals, info