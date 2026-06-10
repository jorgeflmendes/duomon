from __future__ import annotations

from .multi_battle_context import *


class MultiBattleIdentityMixin:
    @property
    def opponent_role(self) -> Optional[str]:
        roles = _multi_opponent_roles(self.player_role)
        return roles[0] if roles else None

    @property
    def partner_role(self) -> Optional[str]:
        return _multi_partner_role(self.player_role)

    @staticmethod
    def _normalize_multi_ident(identifier: str) -> str:
        identifier = str(identifier or "")
        if (
            len(identifier) > 3
            and identifier[0] == "p"
            and identifier[1:2].isdigit()
            and identifier[2:3].isalpha()
            and identifier[3:4] == ":"
        ):
            return identifier[:2] + identifier[3:]
        return identifier

    def get_pokemon(
        self,
        identifier: str,
        force_self_team: bool = False,
        details: str = "",
        request: Optional[Dict[str, Any]] = None,
    ) -> Pokemon:
        normalized = self._normalize_multi_ident(identifier)
        role = normalized[:2]
        same_side = _multi_same_side(role, self.player_role)
        team = self._team if (same_side or force_self_team) else self._opponent_team
        other = self._opponent_team if team is self._team else self._team

        if normalized in team:
            return team[normalized]
        if normalized in other:
            return other[normalized]




        name = self._pokemon_name_from_ident(normalized)
        name_det = str(details or "").split(", ")[0]
        if name_det:
            matches = [
                i
                for i, p in enumerate(team.values())
                if getattr(p, "identifies_as", lambda _: False)(name_det)
            ]
            if len(matches) == 1:
                items = list(team.items())
                old_key, mon = items[matches[0]]
                items[matches[0]] = (normalized, mon)
                try:
                    mon._name = name
                except Exception:
                    pass
                team.clear()
                team.update(dict(items))
                return team[normalized]

        if request:
            team[normalized] = Pokemon(request_pokemon=request, name=name, gen=self.gen)
        elif details:
            team[normalized] = Pokemon(details=details, name=name, gen=self.gen)
        else:
            team[normalized] = Pokemon(species=name, name=name, gen=self.gen)
        return team[normalized]

    @staticmethod
    def _active_from_role(active_dict: Dict[str, Any], role: Optional[str]) -> Optional[Any]:
        if role is None:
            return None
        mon = active_dict.get(role)
        if mon is not None and not bool(safe_getattr(mon, "fainted", False)):
            return mon
        preferred = f"{role}{_multi_slot_letter(role)}"
        mon = active_dict.get(preferred)
        if mon is not None and not bool(safe_getattr(mon, "fainted", False)):
            return mon

        for suffix in ("a", "b", "c"):
            mon = active_dict.get(f"{role}{suffix}")
            if mon is not None and not bool(safe_getattr(mon, "fainted", False)):
                return mon
        for ident, mon in active_dict.items():
            if (
                str(ident).startswith(str(role))
                and mon is not None
                and not bool(safe_getattr(mon, "fainted", False))
            ):
                return mon
        return None

    def _update_team_from_request(self, side: Dict[str, Any], strict_battle_tracking: bool = False):
        if not isinstance(side, dict):
            return

        falsely_active: List[Any] = []
        truly_active: List[Any] = []

        for pokemon in side.get("pokemon", []) or []:
            ident = pokemon.get("ident")
            if not ident:
                continue
            if ident not in self._team:
                self._get_or_create_request_pokemon(pokemon, force_self_team=True)
            if ident not in self._team:



                self._team[ident] = Pokemon(
                    request_pokemon=pokemon,
                    name=self._pokemon_name_from_ident(ident),
                    gen=self.gen,
                )

            mon = self._team.get(ident)
            if mon is None:
                continue
            if pokemon.get("active") and not safe_getattr(mon, "active", False):
                truly_active.append(mon)
            elif not pokemon.get("active") and safe_getattr(mon, "active", False):
                falsely_active.append(mon)

        for illusioned in falsely_active:
            try:
                illusioned.was_illusioned(self.fields)
            except Exception:
                try:
                    illusioned.switch_out(self.fields)
                except Exception:
                    pass

        for illusionist in truly_active:
            try:
                illusionist.switch_in()
            except Exception:
                pass

        for pokemon in side.get("pokemon", []) or []:
            ident = pokemon.get("ident")
            if not ident:
                continue
            mon = self._team.get(ident)
            if mon is None:
                mon = self._get_or_create_request_pokemon(pokemon, force_self_team=True)
            if mon is None:
                continue
            if strict_battle_tracking:
                try:
                    mon.check_consistency(pokemon, self.player_role)
                except Exception:
                    pass
            try:
                mon.update_from_request(pokemon)
            except Exception:
                pass


__all__ = ["MultiBattleIdentityMixin"]
