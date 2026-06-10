from __future__ import annotations

from .multi_battle_context import DoubleBattle
from .multi_battle_identity import MultiBattleIdentityMixin
from .multi_battle_events import MultiBattleEventMixin
from .multi_battle_request import MultiBattleRequestMixin
from .multi_battle_orders import MultiBattleOrderMixin


class MultiBattle(
    MultiBattleIdentityMixin,
    MultiBattleEventMixin,
    MultiBattleRequestMixin,
    MultiBattleOrderMixin,
    DoubleBattle,
):
    pass


__all__ = ["MultiBattle"]
