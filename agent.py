import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

import numpy as np
from utils import ensure_shared_grads

from params import params

def preprocess(obs, gpu_id):
    state = np.concatenate([obs[0], obs[1], obs[2], obs[3]]) / 255.
    state = torch.from_numpy(state).unsqueeze(0).float()

    with torch.cuda.device(gpu_id):
        state = state.cuda()

    return state


def postprocess(logit):
    prob = F.softmax(logit, dim=-1)
    log_prob = F.log_softmax(logit, dim=-1)
    entropy = -(log_prob * prob).sum(1)

    action = prob.multinomial(1).data

    log_prob = log_prob.gather(1, Variable(action))
    action = action.cpu().numpy()

    return action, entropy, log_prob


class run_agent(object):
    def __init__(self, model, gpu_id, goal_storage=None):
        self.model = model
        self.gpu_id = gpu_id
        self.goal_storage = goal_storage
        self.hx, self.cx = None, None
        self.eps_len = 0
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = False
        self.n_update = 0
        self.instruction = None

    def action_train(self, obs, instruction):
        state = preprocess(obs, self.gpu_id)

        target_preds, value, logit, self.hx, self.cx = self.model(state, instruction, self.hx, self.cx)

        action, entropy, log_prob = postprocess(logit)

        self.eps_len += 1
        return np.squeeze(action, axis=0), entropy, value, log_prob

    def action_test(self, observation, instruction):
        with torch.cuda.device(self.gpu_id):
            with torch.no_grad():
                state = preprocess(observation, self.gpu_id)
                target_preds, value, logit, self.hx, self.cx = self.model(state, instruction, self.hx, self.cx)

                prob = F.softmax(logit, dim=-1)
                action = prob.max(1)[1].data.cpu().numpy()

        self.eps_len += 1
        return action

    def synchronize(self, shared_model, instruction):
        self.instruction = instruction

        with torch.cuda.device(self.gpu_id):
            self.model.load_state_dict(shared_model.state_dict())
            self.cx = Variable(torch.zeros(1, 256).cuda())
            self.hx = Variable(torch.zeros(1, 256).cuda())
        self.eps_len = 0

    def put_reward(self, reward, entropy, value, log_prob):
        self.rewards.append(reward)
        self.entropies.append(entropy)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def clear_actions(self):
        self.values.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.entropies.clear()

    def get_reward_len(self):
        return len(self.rewards)

    def training_goal_discriminator(self):
        batch = self.goal_storage.sample(params.goal_batch_size)
        batch_state, label = np.array(batch.goal), np.array(batch.label)

        batch_state = torch.from_numpy(batch_state).float()
        label = torch.tensor(label)

        with torch.cuda.device(self.gpu_id):
            batch_state = Variable(torch.FloatTensor(batch_state)).cuda()
            label = label.cuda()

        target_preds, _, _, _, _ = self.model(batch_state, self.instruction, self.hx, self.cx, use_ActorCritic=False)

        values, indices = target_preds.max(1)
        accuracy = torch.mean((indices.squeeze() == label).float())
        crossentropy_loss = F.cross_entropy(target_preds, label.long())

        return crossentropy_loss, accuracy

    def training(self, next_obs, shared_model, shared_optimizer, rank):
        self.n_update += 1
        self.cx = Variable(self.cx.data)
        self.hx = Variable(self.hx.data)

        R = torch.zeros(1, 1)
        if not self.done:
            state = preprocess(next_obs, self.gpu_id)
            target_preds, value, logit, self.hx, self.cx = self.model(state, self.instruction, self.hx, self.cx)

            R = value.data

        with torch.cuda.device(self.gpu_id):
            R = R.cuda()
        R = Variable(R)
        self.values.append(R)

        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        with torch.cuda.device(self.gpu_id):
            gae = gae.cuda()

        for i in reversed(range(len(self.rewards))):
            R = params.gamma * R + self.rewards[i]
            advantage = R - self.values[i]
            value_loss = value_loss + advantage.pow(2)  # 0.5 *

            # Generalized Advantage Estimation
            delta_t = params.gamma * self.values[i + 1].data - self.values[i].data + self.rewards[i]

            gae = gae * params.gamma * params.tau + delta_t

            policy_loss = policy_loss - self.log_probs[i] * Variable(gae) - params.entropy_coef \
                          * self.entropies[i]

        crossentropy_loss, accuracy = self.training_goal_discriminator()

        shared_optimizer.zero_grad()
        loss = policy_loss + params.value_loss_coef * value_loss + params.goal_coef * crossentropy_loss
        loss.backward()
        clip_grad_norm_(self.model.parameters(), params.clip_grad_norm)
        ensure_shared_grads(self.model, shared_model, gpu=self.gpu_id >= 0)

        shared_optimizer.step()

        if self.n_update % 1000 == 0:
            print("rank: {}, tr_loss: {}, accuracy: {}".format(rank, crossentropy_loss.data, accuracy))

        with torch.cuda.device(self.gpu_id):
            self.model.load_state_dict(shared_model.state_dict())

        self.clear_actions()