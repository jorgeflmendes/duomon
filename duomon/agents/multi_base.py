from __future__ import annotations

from typing import List

from poke_env.data import GenData
from poke_env.exceptions import ShowdownException
from poke_env.teambuilder.teambuilder import Teambuilder

from ..battle.multi_battle import MultiBattle


class MultiAwarePlayerMixin:

    @property
    def format_is_doubles(self) -> bool:
        fmt = str(self.format).lower()
        return "multi" in fmt or "vgc" in fmt or "double" in fmt or "metronome" in fmt

    async def _create_battle(self, split_message: List[str]):
        if "multi" not in str(self.format).lower():
            return await super()._create_battle(split_message)

        if split_message[1] == self._format and len(split_message) >= 2:
            battle_tag = "-".join(split_message)[1:]
            if battle_tag in self._battles:
                return self._battles[battle_tag]

            gen = GenData.from_format(self._format).gen
            battle = MultiBattle(
                battle_tag=battle_tag,
                username=self.username,
                logger=self.logger,
                save_replays=self._save_replays,
                gen=gen,
            )

            if self._current_packed_team:
                battle._teambuilder_team = Teambuilder.parse_packed_team(self._current_packed_team)

            await self._battle_count_queue.put(None)
            if battle_tag in self._battles:
                await self._battle_count_queue.get()
                return self._battles[battle_tag]

            async with self._battle_start_condition:
                self._battle_semaphore.release()
                self._battle_start_condition.notify_all()
                self._battles[battle_tag] = battle

            if self._start_timer_on_battle_start:
                await self.ps_client.send_message("/timer on", battle.battle_tag)

            return battle

        self.logger.critical("Unmanaged battle initialisation message received: %s", split_message)
        raise ShowdownException()


__all__ = ["MultiAwarePlayerMixin"]
