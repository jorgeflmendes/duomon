from __future__ import annotations

from .multi_agent_context import IndependentTwoSlotAgent, MultiAwarePlayerMixin
from .multi_agent_runtime import MultiAgentRuntimeMixin
from .multi_agent_action_filters import MultiAgentActionFilterMixin
from .multi_agent_memory import MultiAgentMemoryMixin
from .multi_agent_decision_gates import MultiAgentDecisionGateMixin
from .multi_agent_comm_facts import MultiAgentCommFactsMixin
from .multi_agent_comm_proposals import MultiAgentCommProposalMixin
from .multi_agent_comm_runtime import MultiAgentCommRuntimeMixin
from .multi_agent_joint_selection import MultiAgentJointSelectionMixin
from .multi_agent_joint_scoring import MultiAgentJointScoringMixin
from .multi_agent_hard_values import MultiAgentHardValueMixin
from .multi_agent_response_trade import MultiAgentResponseTradeMixin
from .multi_agent_response_prediction import MultiAgentResponsePredictionMixin
from .multi_agent_summaries import MultiAgentSummaryMixin


class SingleSlotMultiAgent(
    MultiAgentRuntimeMixin,
    MultiAgentActionFilterMixin,
    MultiAgentMemoryMixin,
    MultiAgentDecisionGateMixin,
    MultiAgentCommFactsMixin,
    MultiAgentCommProposalMixin,
    MultiAgentCommRuntimeMixin,
    MultiAgentJointSelectionMixin,
    MultiAgentJointScoringMixin,
    MultiAgentHardValueMixin,
    MultiAgentResponseTradeMixin,
    MultiAgentResponsePredictionMixin,
    MultiAgentSummaryMixin,
    MultiAwarePlayerMixin,
    IndependentTwoSlotAgent,
):
    pass


__all__ = ["SingleSlotMultiAgent"]
