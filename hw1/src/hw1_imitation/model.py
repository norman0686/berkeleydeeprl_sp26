"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from einops import pack, rearrange
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        self.chunk_size = chunk_size
        self.state_dim = state_dim
        self.action_dim = action_dim

        in_dim = state_dim
        out_dim = action_dim * chunk_size

        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred_action_chunk = rearrange(
            self.mlp(state),
            "batch (chunk action) -> batch chunk action",
            chunk=self.chunk_size,
        )
        loss = torch.mean((pred_action_chunk - action_chunk) ** 2)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        with torch.no_grad():
            pred_action_chunk = rearrange(
                self.mlp(state),
                "batch (chunk action) -> batch chunk action",
                chunk=self.chunk_size,
            )
        return pred_action_chunk


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        self.chunk_size = chunk_size
        self.state_dim = state_dim
        self.action_dim = action_dim

        in_dim = state_dim + action_dim * chunk_size + 1
        out_dim = action_dim * chunk_size

        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        tau = torch.rand(batch_size, 1, 1, device=state.device, dtype=state.dtype)
        noise = torch.randn_like(action_chunk)
        interpolation = tau * action_chunk + (1 - tau) * noise
        target_velocity = action_chunk - noise

        model_input, _ = pack(
            [state, interpolation, rearrange(tau, "... 1 -> ...")], "batch *"
        )
        model_output = self.mlp(model_input)
        predicted_velocity = rearrange(
            model_output,
            "batch (chunk action) -> batch chunk action",
            chunk=self.chunk_size,
        )
        loss = torch.mean((predicted_velocity - target_velocity) ** 2)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        with torch.no_grad():
            batch_size = state.shape[0]
            action_chunk = torch.randn(
                batch_size,
                self.chunk_size,
                self.action_dim,
                device=state.device,
                dtype=state.dtype,
            )
            for i in range(num_steps):
                tau = i / num_steps
                packed_input, _ = pack(
                    [
                        state,
                        action_chunk,
                        torch.full(
                            (batch_size, 1), tau, device=state.device, dtype=state.dtype
                        ),
                    ],
                    "batch *",
                )
                output = self.mlp(packed_input)
                predicted_velocity = rearrange(
                    output,
                    "batch (chunk action) -> batch chunk action",
                    chunk=self.chunk_size,
                )
                action_chunk = action_chunk + predicted_velocity / num_steps

        return action_chunk


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
