import logging
from pathlib import Path
from .action_decoder import ActionDecoder
import numpy as np
import skill_generator.models.skill_generator as model_sg
# import skill_generator.skill_generator.models.skill_generator as model_sg
from typing import Optional, Tuple
import spil
import hydra
from omegaconf import DictConfig
from spil.utils.utils import get_last_checkpoint
import torch
import torch.nn as nn
from spil.models.decoders.utils.gripper_control import tcp_to_world_frame, world_to_tcp_frame
from torch.distributions import Normal
from collections import deque
from spil.models.decoders.utils.rnn import gru_decoder, lstm_decoder, mlp_decoder, rnn_decoder  # needed for line 60

logger = logging.getLogger(__name__)


class SkillDecoder(ActionDecoder):
    def __init__(
            self,
            perceptual_features: int,
            latent_goal_features: int,
            lang_in_features: int,
            plan_features: int,
            hidden_size: int,
            hidden_size2: int,
            out_features: int,
            policy_rnn_dropout_p: float,
            criterion: str,
            num_layers: int,
            rnn_model: str,
            perceptual_emb_slice: list, 
            gripper_control: bool,
            sg_chk_path: str,
            skill_len: int,
            skill_num: int,
            gamma_1: float,
            gamma_2: float,
            base_skill_prior: list
    ):
        super(SkillDecoder, self).__init__()
        self.plan_features = plan_features
        self.gripper_control = gripper_control
        self.out_features = out_features
        in_features = (perceptual_emb_slice[1] - perceptual_emb_slice[0]) + latent_goal_features + plan_features
        self.rnn = eval(rnn_model)
        self.rnn = self.rnn(in_features, hidden_size, num_layers, policy_rnn_dropout_p)
        self.skills = nn.Linear(hidden_size, out_features)
        self.skill_len = skill_len
        self.skill_selector = eval(rnn_model)
        self.skill_selector = self.skill_selector((perceptual_emb_slice[1] - perceptual_emb_slice[0]) + lang_in_features, hidden_size2, num_layers, policy_rnn_dropout_p)
        self.skill_classes = nn.Sequential(
            nn.Linear(hidden_size2, skill_num),
            nn.Softmax(dim=-1)
        )
        
        self.action_dim = 7 #  delta XYZ position, delta euler angles and the gripper action.
        self.max_plan_timesteps = 30 # the maximum timesteps of each plan
        self.plan_actions = torch.zeros([self.max_plan_timesteps, self.max_plan_timesteps+self.skill_len, self.action_dim]).cuda() 

        self.criterion = getattr(nn, criterion)()
        self.perceptual_emb_slice = perceptual_emb_slice # [0, 128]
        self.time_slice = [0, None, skill_len] # [0, None, 5]
        self.skill_num = skill_num # 3
        self.hidden_state = {'skill_emb': None, 'skill_cls': None}
        self.hidden_state2 = list()
        for i in range(skill_len):
            self.hidden_state2.append({'skill_emb': None, 'skill_cls': None})
        self.sg_chk_path = Path(sg_chk_path)
        logger.info(f'load from skill generator checkpoint {self.sg_chk_path}')
        if not self.sg_chk_path.is_absolute():
            self.sg_chk_path = Path(spil.__file__).parents[1] / self.sg_chk_path

        self.skill_len = torch.tensor(skill_len)
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.base_skill_prior = torch.tensor(base_skill_prior)
        self.cached_actions = deque([])
        self._load_checkpoint()

    def _load_checkpoint(self):
        """

        load the checkpoint of skill generator. Here, we need the model action_decoder and prior_locator module in the skill generator

        """
        chk = get_last_checkpoint(self.sg_chk_path)
        logger.info(f'load from skill generator checkpoint {chk}')
        if chk is not None:
            self.skill_generator = getattr(model_sg, 'SkillGenerator').load_from_checkpoint(chk.as_posix())
        self.skill_generator.freeze()
        self.action_decoder = self.skill_generator.decoder.eval()
        self.prior_locator = self.skill_generator.prior_locator.eval()

    def _get_skill_emb(self, latent_plan: torch.Tensor, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor, h_0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([latent_plan, perceptual_emb, latent_goal], dim=-1)  # b, s, (plan + visuo-propio + goal)
        if not isinstance(self.rnn, nn.Sequential) and isinstance(self.rnn, nn.RNNBase):
            x, h_n = self.rnn(x, h_0)
        else:
            x = self.rnn(x)
            h_n = None
        skill_emb = self.skills(x)
        return skill_emb, h_n

    def _get_skill_cls(self, lang_emb: torch.Tensor, perceptual_emb: torch.Tensor, h_0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lang_emb: the embedding contains the language information. It is used to construct sequences of skill types are the prior knowledge for skill sampling.
            perceptual_emb: the start perceptual_emb of the first frame
            h_0: the initial hidden state for skill classes generation

        Returns:

        """
        x = torch.cat([perceptual_emb, lang_emb], dim=-1)
        if not isinstance(self.skill_selector, nn.Sequential) and isinstance(self.skill_selector, nn.RNNBase):
            x, h_n = self.skill_selector(x, h_0)
        else:
            x = self.skill_selector(x)
            h_n = None
        skill_cls = self.skill_classes(x)
        return skill_cls, h_n

    def clear_hidden_state(self) -> None:
        self.hidden_state = {'skill_emb': None, 'skill_cls': None}
        
        self.hidden_state2 = list()
        for i in range(self.skill_len):
            self.hidden_state2.append({'skill_emb': None, 'skill_cls': None})

    def clear_cached_actions(self) -> None:
        self.cached_actions = deque([])
        self.cached_actions2 = list()
        for i in range(self.skill_len):
            self.cached_actions2.append(deque([]))

    def forward(
            self,
            latent_plan: torch.Tensor,
            perceptual_emb: torch.Tensor,
            latent_goal: torch.Tensor,
            lang_emb: torch.Tensor = None,
            hs_0: Optional[torch.Tensor] = None,
            hc_0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent_plan: plan_embedding in the latent space with the shape of (B, N_z)
            perceptual_emb: perceptual_embedding with the shape of (B, T, N_p), T: time sequence
            latent_goal: goal_embedding with the shape of (B, N_g)
            lang_emb: the embedding contains the language information. It is used to reconstruct sequences of skill types that are the prior knowledge for skill sampling.
            hs_0: the initial hidden states for skill embedding sequences
            hc_0: the initial hidden states for skill class sequences

        Returns:
            skill_emb: the sequence of skill embeddings with the shape of (B, T, N_s)
            hs_n: the output hidden states for skill embedding sequences
            skill_cls: the sequence of skill classes with the shape of (B, T, 3)
            hc_n: the output hidden states for skill class sequences
            act_seq_len: the required action sequence length
        """
        act_seq_len = perceptual_emb.shape[1] # sequence length of the action, T steps
        perceptual_emb = perceptual_emb[..., slice(*self.time_slice), slice(*self.perceptual_emb_slice)]
        batch_size, skill_seq_len = perceptual_emb.shape[0], perceptual_emb.shape[1]
        latent_plan = latent_plan.unsqueeze(1).expand(-1, skill_seq_len, -1)
        latent_goal = latent_goal.unsqueeze(1).expand(-1, skill_seq_len, -1)

        skill_emb, hs_n = self._get_skill_emb(latent_plan, perceptual_emb, latent_goal, hs_0)
        if lang_emb is not None:
            lang_emb = lang_emb.unsqueeze(1).expand(-1, skill_seq_len, -1)
            skill_cls, hc_n = self._get_skill_cls(lang_emb,  perceptual_emb, hc_0)
        else:
            skill_cls, hc_n = None, None
        return skill_emb, hs_n, skill_cls, hc_n, act_seq_len

    def _action_generation(
            self,
            skill_emb: torch.Tensor,
            act_seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            skill_emb: a sequence of skill embeddings with the shape of (B, T_s, N_s)
            act_seq_len: the required action sequence length

        Returns:
            pred_actions: the predicted action sequence with the shape of (B, T_a, N_s)
        """
        (B, T_s, N_s) = skill_emb.shape
        skill_emb = torch.flatten(skill_emb, start_dim=0, end_dim=1)  # (B * T_s, N_s)
        actions = self.action_decoder(skill_emb, self.skill_len.repeat(B * T_s))  # (B * T_s, T_a, N_a)
        (_, T_a, N_a) = actions.shape
        pred_actions = actions.reshape(B, T_s, T_a, N_a)
        pred_actions = torch.flatten(pred_actions, start_dim=1, end_dim=2)
        if act_seq_len is not None:
            pred_actions = pred_actions[:, :act_seq_len, :]
        return pred_actions

    def _base_skill_reg_loss(
            self,
            skill_emb: torch.Tensor,
            skill_cls: torch.Tensor,

    ):
        """

        Args:
            skill_cls: the skill selection results with the shape of (B, T, 3)
            skill_emb: the skill embedding in the latent space with the shape of (B, T, N)
        Returns:
            reg_loss: the loss to regularize the skills based on the selected base skills
        """
        B, T, _ = skill_cls.shape
        priors = self.prior_locator(repeat=B * T)  # (B*T, 3, N)
        dist = [Normal(priors['p_mu'][:, i, :].reshape(B, T, -1), priors['p_scale'][:, i, :].reshape(B, T, -1)) for i in range(self.skill_num)]
        nll = [-1 * d.log_prob(skill_emb).mean(dim=-1) for d in dist]
        nll = torch.stack(nll, dim=-1)
        return torch.sum(nll * skill_cls, dim=-1).mean()

    def _categorical_reg_loss(
            self,
            skill_cls: torch.Tensor,
            eps: float = 1e-3
    ):
        """
        Args:
            skill_cls: the skill labelling results with the shape of (B, T, 3)
        Returns:
            categorical_reg_loss: the loss to regularize the base skill selection.
        """
        skill_cls = torch.clip(skill_cls, min=eps)
        return torch.mean(skill_cls * torch.log(skill_cls / self.base_skill_prior[None, None, :].to(skill_cls)))

    def act(
            self,
            latent_plan: torch.Tensor,
            perceptual_emb: torch.Tensor,
            latent_goal: torch.Tensor,
            robot_obs: Optional[torch.Tensor] = None,
            lang_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        The cached_actions are used to store the predicted action sequence (5 actions) generated from the skill decoder. 
        Each step return the first action in the cached actions and remove it from the deque, until the deque is empty.
        The the skill decoder will be called to generate the next action sequence based on the current state.
        
        Args:
            latent_plan: the plan embedding in the latent space with the shape of (B, N_z)
            perceptual_emb: the perceptual embedding with the shape of (B, T, N_p)
            latent_goal: the goal embedding with the shape of (B, N_g)
            robot_obs: the current observation of the robot which is used for gripper frame and world frame transformation
            lang_emb: the embeddings contains language information with the shape of (B, N_l)

        Returns:
            pred_actions: the predicted actions with the shape of (B, T_a, N_a)
        """
        if not self.cached_actions: # cached_actions is a deque
            skill_emb, self.hidden_state['skill_emb'], skill_cls, self.hidden_state['skill_cls'], act_seq_len = self(
                latent_plan, perceptual_emb, latent_goal,
                hs_0=self.hidden_state['skill_emb'], hc_0=self.hidden_state['skill_cls'])
            pred_actions = self._action_generation(skill_emb)
            if self.gripper_control:
                pred_actions_world = tcp_to_world_frame(pred_actions, robot_obs)
                for i in range(pred_actions.shape[1]):
                    self.cached_actions.append(pred_actions_world[:, i:i+1, :])
            else:
                for i in range(pred_actions.shape[1]):
                    self.cached_actions.append(pred_actions[:, i:i+1, :])

        return self.cached_actions.popleft()

    def act_chunk_with_queue(
            self,
            latent_plan: torch.Tensor,
            perceptual_emb: torch.Tensor,
            latent_goal: torch.Tensor,
            robot_obs: Optional[torch.Tensor] = None,
            lang_emb: Optional[torch.Tensor] = None,
            rollout_step: Optional[int] = 0,
    ) -> torch.Tensor:
        """
        The cached_actions are used to store the predicted action sequence (5 actions) generated from the skill decoder. 
        Each step return the first action in the cached actions and remove it from the deque, until the deque is empty.
        The the skill decoder will be called to generate the next action sequence based on the current state.
        
        Args:
            latent_plan: the plan embedding in the latent space with the shape of (B, N_z)
            perceptual_emb: the perceptual embedding with the shape of (B, T, N_p)
            latent_goal: the goal embedding with the shape of (B, N_g)
            robot_obs: the current observation of the robot which is used for gripper frame and world frame transformation
            lang_emb: the embeddings contains language information with the shape of (B, N_l)

        Returns:
            pred_actions: the predicted actions with the shape of (B, T_a, N_a)
        """
        p = rollout_step % self.skill_len
        skill_emb, self.hidden_state2[p]['skill_emb'], skill_cls, self.hidden_state2[p]['skill_cls'], act_seq_len = self(
            latent_plan, perceptual_emb, latent_goal,
            hs_0=self.hidden_state2[p]['skill_emb'], hc_0=self.hidden_state2[p]['skill_cls'])
        pred_actions = self._action_generation(skill_emb)
        if self.gripper_control:
            pred_actions = tcp_to_world_frame(pred_actions, robot_obs) # (B, T_a, N_a) = [1, 5, 7]  
        
        time = rollout_step % self.max_plan_timesteps
        if time == 0:
            self.plan_actions = torch.zeros([self.max_plan_timesteps, self.max_plan_timesteps+self.skill_len, self.action_dim]).cuda()
        
        for i in range(pred_actions.shape[1]):
            self.plan_actions[time, time+i:time+i+1, :] = pred_actions[:, i:i+1, :]
        
        index = (time // self.skill_len) * self.skill_len
        curr_step_action = self.plan_actions[index:time+1,time]
        # k = 1 ## average len = 1.314, 1.24
        # k = 0.9 ## average 
        # k = 1.1 # len = 1.27
        k = 0.01
        weights = np.exp(-k * np.arange(len(curr_step_action)))
        # weights = np.ones(len(curr_step_action))
        weights = weights / np.sum(weights)
        weights = torch.from_numpy(weights).float().cuda().unsqueeze(dim=1)
        action = (curr_step_action * weights).sum(dim=0, keepdim=True)
        
        return action

    # Predict the action a_t based on the current state s_t, a_t = sum(w_i * S_i(t-i)), i = 1, 2, ..., t
    # S_i(t) is the action in the i-th action sequence at the current time t, w_i is the weight of the action
    # i_1,        S_1: (a_11, a_12, a_13, a_14, a_15),             S_1(3-1) = S_1(2) = a_13
    # i_2,        S_2: _____ (a_21, a_22, a_23, a_24, a_25),       S_2(3-2) = S_2(1) = a_22
    # i_3,        S_3: ____________(a_31, a_32, a_33, a_34, a_35), S_3(t-i) = S_3(3-3) = a_31
    # a_3 = w_1 * a_13 + w_2 * a_22 + w_3 * a_31
    # Temporal ensembling idea from Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware
    # Need test
    def act_t(
            self,
            latent_plan: torch.Tensor,
            perceptual_emb: torch.Tensor,
            latent_goal: torch.Tensor,
            robot_obs: Optional[torch.Tensor] = None,
            lang_emb: Optional[torch.Tensor] = None,
            rollout_step: Optional[int] = 0,
    ) -> torch.Tensor:
        """
        The cached_actions are used to store the predicted action sequence (5 actions) generated from the skill decoder. 
        Return the first action in the cached_actions at each step and remove it from the deque, until the deque is empty.
        If the cached_actions is empty, the next action sequence will be stored in the deque based on the current state.
        
        Args:
            latent_plan: the plan embedding in the latent space with the shape of (B, N_z)
            perceptual_emb: the perceptual embedding with the shape of (B, T, N_p)
            latent_goal: the goal embedding with the shape of (B, N_g)
            robot_obs: the current observation of the robot which is used for gripper frame and world frame transformation
            lang_emb: the embeddings contains language information with the shape of (B, N_l)

        Returns:
            pred_actions: the predicted actions with the shape of (B, T_a, N_a), N_a = 7
            return one action at the current time step
        """
        skill_emb, self.hidden_state['skill_emb'], skill_cls, self.hidden_state['skill_cls'], act_seq_len = self(
                latent_plan, perceptual_emb, latent_goal,
                hs_0=self.hidden_state['skill_emb'], hc_0=self.hidden_state['skill_cls'])
        pred_actions = self._action_generation(skill_emb)
        if self.gripper_control:
            pred_actions = tcp_to_world_frame(pred_actions, robot_obs)
        
        time = rollout_step % self.max_plan_timesteps
        if time == 0:
            self.plan_actions = torch.zeros([self.max_plan_timesteps, self.max_plan_timesteps+self.skill_len, self.action_dim]).cuda()
        
        for i in range(pred_actions.shape[1]):
            self.plan_actions[time, time+i:time+i+1, :] = pred_actions[:, i:i+1, :]
        
        curr_step_action = self.plan_actions[:,time,:]
        select_action = torch.all(curr_step_action != 0, axis=1)
        curr_step_action = curr_step_action[select_action]
        k = 0.01
        weights = np.exp(-k * np.arange(len(select_action)))
        weights = weights / np.sum(weights)
        weights = torch.from_numpy(weights).float().cuda().unsqueeze(dim=1)
        action = (curr_step_action * weights).sum(dim=0, keepdim=True)
        
        return action

    @staticmethod
    def _hinge_loss(pred_gripper_actions, gt_gripper_actions, eps=1e-6):
        return torch.clamp(1.0 - pred_gripper_actions * gt_gripper_actions, min=eps).mean() - eps

    # calculate the loss of the predicted actions and the ground truth actions, including the hinge loss for gripper control
    def _loss(self, pred_actions, gt_actions):
        loss = self.criterion(pred_actions[..., :6], gt_actions[..., :6])
        hinge_loss = self._hinge_loss(pred_actions[..., 6], gt_actions[..., 6])
        return 0.85 * loss + 0.15 * hinge_loss

    def loss_and_act(
            self,
            latent_plan: torch.Tensor,
            perceptual_emb: torch.Tensor,
            latent_goal: torch.Tensor,
            actions: torch.Tensor,
            robot_obs: Optional[torch.Tensor] = None,
            lang_emb: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent_plan: the plan embedding in the latent space with the shape of (B, N_z)
            perceptual_emb: the perceptual embedding with the shape of (B, T, N_p)
            latent_goal: the goal embedding with the shape of (B, N_g)
            actions: the ground truth actions with the shape of (B, T_a, N_a)
            robot_obs: the current observation of the robot which is used for gripper frame and world frame transformation
            lang_emb: the embeddings contains language information with the shape of (B, N_l)

        Returns:
            loss: the reconstruction and regularization loss for optimization
            acts: the predicted actions sequence
            reg_loss: the scaled reconstruction loss which regularize the skill bases
        """

        skill_emb, _, skill_cls, _, act_seq_len = self(
            latent_plan,
            perceptual_emb,
            latent_goal,
            lang_emb = lang_emb
        )
        pred_actions = self._action_generation(skill_emb, act_seq_len)
        if skill_cls is not None:
            base_skill_reg_loss = self._base_skill_reg_loss(skill_emb, skill_cls)
            categorical_reg_loss = self._categorical_reg_loss(skill_cls)
        else:
            base_skill_reg_loss = 0.
            categorical_reg_loss = 0.
        # loss
        if self.gripper_control:
            actions_tcp = world_to_tcp_frame(actions, robot_obs)
            loss = self._loss(pred_actions, actions_tcp) 
            pred_actions_world = tcp_to_world_frame(pred_actions, robot_obs)
            return loss + self.gamma_1 * base_skill_reg_loss + self.gamma_2 * categorical_reg_loss, pred_actions_world
        else:
            loss = self._loss(pred_actions, actions)
            return loss + self.gamma_1 * base_skill_reg_loss + self.gamma_2 * categorical_reg_loss, pred_actions

    def loss(
            self,
            latent_plan: torch.Tensor,
            perceptual_emb: torch.Tensor,
            latent_goal: torch.Tensor,
            actions: torch.Tensor,
            robot_obs: Optional[torch.Tensor] = None,
            lang_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        skill_emb, _, skill_cls, _, act_seq_len = self(
            latent_plan,
            perceptual_emb,
            latent_goal,
            lang_emb = lang_emb
        )
        pred_actions = self._action_generation(skill_emb, act_seq_len)
        if skill_cls is not None:
            base_skill_reg_loss = self._base_skill_reg_loss(skill_emb, skill_cls)
            categorical_reg_loss = self._categorical_reg_loss(skill_cls)
        else:
            base_skill_reg_loss = 0.
            categorical_reg_loss = 0.
        if self.gripper_control:
            actions_tcp = world_to_tcp_frame(actions, robot_obs)
            loss = self._loss(pred_actions, actions_tcp) 
            return loss + self.gamma_1 * base_skill_reg_loss + self.gamma_2 * categorical_reg_loss
        else:
            loss = self._loss(pred_actions, actions)
            return loss + self.gamma_1 * base_skill_reg_loss + self.gamma_2 * categorical_reg_loss
