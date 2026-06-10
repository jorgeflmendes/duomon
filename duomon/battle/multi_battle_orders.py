from __future__ import annotations

from .multi_battle_context import *


class MultiBattleOrderMixin:
    def _get_target_mon(
        self, pokemon: str, target_type: str, target_str: Optional[str]
    ) -> Optional[Any]:
        if target_str is not None and _multi_same_side(pokemon[:2], target_str[:2]):
            return None
        if target_type != "all" and target_str is not None:
            return self.get_pokemon(target_str)
        targets = (
            self.opponent_active_pokemon
            if _multi_same_side(self.player_role, pokemon[:2])
            else self.active_pokemon
        )
        for target in targets:
            if target is not None and safe_getattr(target, "ability", None) == "pressure":
                return target
        return None

    def switch(self, pokemon_str: str, details: str, hp_status: str):
        self._debug_multi_state_event(
            "switch_begin",
            {"pokemon": pokemon_str, "details": details, "hp_status": hp_status},
        )
        pokemon_identifier = pokemon_str.split(":")[0][:3]
        player_identifier = pokemon_identifier[:2]
        team = self._active_team_for_role(player_identifier)
        pokemon_out = team.pop(pokemon_identifier, None)
        if pokemon_out is not None:
            pokemon_out.switch_out(self.fields)
            if safe_species(pokemon_out).lower() == "dondozo":
                try:
                    self._clear_commander_from_partner(pokemon_identifier)
                except Exception:
                    pass
        pokemon_in = self.get_pokemon(
            pokemon_str,
            force_self_team=_multi_same_side(player_identifier, self.player_role),
            details=details,
        )
        pokemon_in.switch_in()
        pokemon_in.set_hp_status(hp_status)
        team[pokemon_identifier] = pokemon_in
        self._debug_multi_state_event(
            "switch_end",
            {"pokemon": pokemon_str, "details": details, "hp_status": hp_status},
        )

    def end_illusion(self, pokemon_name: str, details: str):
        player_identifier = pokemon_name[:2]
        pokemon_identifier = pokemon_name[:3]
        active_dict = self._active_team_for_role(player_identifier)
        active = active_dict.get(pokemon_identifier)


        if _multi_same_side(player_identifier, self.player_role):
            self.get_pokemon(pokemon_name, force_self_team=True, details=details)
        if active is None:
            try:
                pokemon = self.get_pokemon(
                    pokemon_name,
                    force_self_team=_multi_same_side(player_identifier, self.player_role),
                    details=details,
                )
                pokemon.switch_in()
                active_dict[pokemon_identifier] = pokemon
            except Exception:
                pass
            return
        try:
            active_dict[pokemon_identifier] = self._end_illusion_on(
                illusioned=active,
                illusionist=pokemon_name,
                details=details,
            )
        except ValueError:
            pokemon = self.get_pokemon(
                pokemon_name,
                force_self_team=_multi_same_side(player_identifier, self.player_role),
                details=details,
            )
            pokemon.switch_in()
            active_dict[pokemon_identifier] = pokemon

    def _swap(self, pokemon_str: str, slot: str):


        return

    @property
    def active_pokemon(self) -> List[Optional[Any]]:
        if self.player_role is None:
            raise ValueError("Unable to get active_pokemon, player_role is None")
        own = self._active_from_role(self._active_pokemon, self.player_role)
        partner = self._active_from_role(self._active_pokemon, self.partner_role)
        return [own, partner]

    @property
    def opponent_active_pokemon(self) -> List[Optional[Any]]:
        roles = _multi_opponent_roles(self.player_role)
        opponents = [self._active_from_role(self._opponent_active_pokemon, role) for role in roles]
        if any(mon is None for mon in opponents):
            for idx, role in enumerate(roles):
                if idx >= len(opponents) or opponents[idx] is not None:
                    continue
                for ident, mon in (safe_getattr(self, "_opponent_team", {}) or {}).items():
                    if (
                        str(ident).startswith(str(role))
                        and mon is not None
                        and not bool(safe_getattr(mon, "fainted", False))
                    ):
                        opponents[idx] = mon
                        break
        return (opponents + [None, None])[:2]

    def get_possible_showdown_targets(
        self, move: Any, pokemon: Any, dynamax: bool = False
    ) -> List[int]:
        mid = move_id(move)
        target_type = move_target_type(move)

        if mid in PROTECT_MOVES or is_spread_move(move) or target_type in NO_EXPLICIT_TARGETS:
            return [self.EMPTY_TARGET_POSITION]

        if target_type in {
            "self",
            "allySide",
            "foeSide",
            "field",
            "all",
            "allAdjacent",
            "allAdjacentFoes",
            "allyTeam",
        }:
            return [self.EMPTY_TARGET_POSITION]

        if target_type in {"adjacentAlly", "adjacentAllyOrSelf"}:
            targets = []
            if self.active_pokemon[1] is not None:
                targets.append(self.POKEMON_2_POSITION)
            if target_type == "adjacentAllyOrSelf":
                targets.append(self.POKEMON_1_POSITION)
            return targets or [self.EMPTY_TARGET_POSITION]

        targets: List[int] = []
        if self.opponent_active_pokemon[0] is not None:
            targets.append(self.OPPONENT_1_POSITION)
        if self.opponent_active_pokemon[1] is not None:
            targets.append(self.OPPONENT_2_POSITION)
        if target_type in {"any", "normal"} and self.active_pokemon[1] is not None:

            targets.append(self.POKEMON_2_POSITION)
        return targets or [self.EMPTY_TARGET_POSITION]

    def to_showdown_target(self, move: Any, target_mon: Optional[Any]) -> int:
        if target_mon is None:
            return self.EMPTY_TARGET_POSITION
        if target_mon == self.active_pokemon[0]:
            return self.POKEMON_1_POSITION
        if target_mon == self.active_pokemon[1]:
            return self.POKEMON_2_POSITION
        if target_mon == self.opponent_active_pokemon[0]:
            return self.OPPONENT_1_POSITION
        if target_mon == self.opponent_active_pokemon[1]:
            return self.OPPONENT_2_POSITION
        return self.EMPTY_TARGET_POSITION

    @property
    def valid_orders(self) -> List[List[SingleBattleOrder]]:


        try:
            orders = super().valid_orders
            return [orders[0] or [DefaultBattleOrder()], [PassBattleOrder()]]
        except Exception:
            return [[DefaultBattleOrder()], [PassBattleOrder()]]


__all__ = ["MultiBattleOrderMixin"]
