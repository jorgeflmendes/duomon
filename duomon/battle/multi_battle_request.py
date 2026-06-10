from __future__ import annotations

from .multi_battle_context import *


class MultiBattleRequestMixin:
    def parse_request(self, request: Dict[str, Any], strict_battle_tracking: bool = False):




        self._last_raw_request = request if isinstance(request, dict) else {}
        self._debug_multi_state_event(
            "parse_request_begin",
            {
                "rqid": self._last_raw_request.get("rqid"),
                "wait": self._last_raw_request.get("wait"),
                "active_len": len(self._last_raw_request.get("active") or []),
                "side_active": self._debug_request_active_idents(
                    self._last_raw_request.get("side")
                ),
                "ally_active": self._debug_request_active_idents(
                    self._last_raw_request.get("ally")
                ),
            },
        )






        self._seed_multi_request_side(request.get("side"), force_self_team=True)
        self._seed_multi_request_side(request.get("ally"), force_self_team=True)



        super().parse_request(request, strict_battle_tracking=False)
        self._normalize_multi_active_ownership()
        self._remap_own_active_slot()
        self._debug_multi_state_event(
            "parse_request_after_base", {"rqid": self._last_raw_request.get("rqid")}
        )




        self._ingest_multi_ally_request(request)
        self._normalize_multi_active_ownership()
        self._debug_multi_state_event(
            "parse_request_after_ally", {"rqid": self._last_raw_request.get("rqid")}
        )




        if "active" in request and len(request.get("active", [])) <= 1:
            self._available_moves = [
                (self._available_moves[0] if self._available_moves else []),
                [],
            ]
            self._available_switches = [
                (self._available_switches[0] if self._available_switches else []),
                [],
            ]
            self._can_dynamax = [(self._can_dynamax[0] if self._can_dynamax else False), False]
            self._can_mega_evolve = [
                (self._can_mega_evolve[0] if self._can_mega_evolve else False),
                False,
            ]
            self._can_tera = [(self._can_tera[0] if self._can_tera else False), False]
            self._can_z_move = [(self._can_z_move[0] if self._can_z_move else False), False]
            force = (
                self._force_switch
                if isinstance(self._force_switch, list)
                else [bool(self._force_switch)]
            )
            trapped = self._trapped if isinstance(self._trapped, list) else [bool(self._trapped)]
            maybe_trapped = (
                self._maybe_trapped
                if isinstance(self._maybe_trapped, list)
                else [bool(self._maybe_trapped)]
            )






            raw_force = request.get("forceSwitch") if isinstance(request, dict) else None
            if isinstance(raw_force, list):
                own_force = any(bool(x) for x in raw_force)
            elif raw_force is not None:
                own_force = bool(raw_force)
            else:
                own_force = bool(force[0]) if force else False

            self._force_switch = [own_force, False]
            self._trapped = [(trapped[0] if trapped else False), False]
            self._maybe_trapped = [(maybe_trapped[0] if maybe_trapped else False), False]

        self._debug_multi_state_event(
            "parse_request_end", {"rqid": self._last_raw_request.get("rqid")}
        )

    def _normalize_multi_active_ownership(self) -> None:
        role = safe_getattr(self, "player_role", None)
        if role is None:
            return

        for source in (self._active_pokemon, self._opponent_active_pokemon):
            for ident, mon in list(source.items()):
                mon_role = str(ident or "")[:2]
                if len(mon_role) != 2 or mon_role[0] != "p" or not mon_role[1].isdigit():
                    continue
                correct = (
                    self._active_pokemon
                    if _multi_same_side(mon_role, role)
                    else self._opponent_active_pokemon
                )
                if correct is source:
                    continue
                existing = correct.get(ident)
                if existing is None or bool(safe_getattr(existing, "fainted", False)):
                    correct[ident] = mon
                source.pop(ident, None)

        for source in (self._team, self._opponent_team):
            for ident, mon in list(source.items()):
                mon_role = str(ident or "")[:2]
                if len(mon_role) != 2 or mon_role[0] != "p" or not mon_role[1].isdigit():
                    continue
                correct = self._team if _multi_same_side(mon_role, role) else self._opponent_team
                if correct is source:
                    continue
                if ident not in correct:
                    correct[ident] = mon
                source.pop(ident, None)

    @staticmethod
    def _debug_request_active_idents(side: Optional[Dict[str, Any]]) -> List[str]:
        if not isinstance(side, dict):
            return []
        return [
            str(pokemon.get("ident") or "")
            for pokemon in (side.get("pokemon", []) or [])
            if isinstance(pokemon, dict) and pokemon.get("active")
        ]

    @staticmethod
    def _debug_pokemon_summary(mon: Any) -> Dict[str, Any]:
        if mon is None:
            return {"species": "none"}
        return {
            "species": safe_species(mon),
            "hp": round(float(safe_hp_fraction(mon)), 4),
            "fainted": bool(safe_getattr(mon, "fainted", False)),
        }

    def _debug_multi_state_event(
        self, event: str, payload: Optional[Dict[str, Any]] = None
    ) -> None:
        path = os.environ.get("DUOMON_MULTI_STATE_DEBUG_PATH", "").strip()
        if not path:
            return
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            record = {
                "ts": time.time(),
                "event": event,
                "battle_tag": safe_getattr(self, "battle_tag", "unknown"),
                "role": safe_getattr(self, "player_role", None),
                "turn": int(safe_getattr(self, "turn", 0) or 0),
                "payload": payload or {},
                "active_keys": sorted(
                    str(k) for k in (safe_getattr(self, "_active_pokemon", {}) or {}).keys()
                ),
                "opponent_active_keys": sorted(
                    str(k)
                    for k in (safe_getattr(self, "_opponent_active_pokemon", {}) or {}).keys()
                ),
                "team_keys": sorted(str(k) for k in (safe_getattr(self, "_team", {}) or {}).keys()),
                "opponent_team_keys": sorted(
                    str(k) for k in (safe_getattr(self, "_opponent_team", {}) or {}).keys()
                ),
                "active": [self._debug_pokemon_summary(mon) for mon in self.active_pokemon],
                "opponent_active": [
                    self._debug_pokemon_summary(mon) for mon in self.opponent_active_pokemon
                ],
            }
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(json_safe(record), ensure_ascii=True) + "\n")
        except Exception:
            return

    @staticmethod
    def _pokemon_name_from_ident(identifier: str) -> str:
        return str(identifier[4:] if len(identifier) > 4 else identifier).strip()

    def _seed_multi_request_side(
        self, side: Optional[Dict[str, Any]], force_self_team: bool
    ) -> None:
        if not isinstance(side, dict):
            return
        for pokemon in side.get("pokemon", []) or []:
            identifier = pokemon.get("ident")
            if not identifier:
                continue
            self._get_or_create_request_pokemon(pokemon, force_self_team=force_self_team)

    def _get_or_create_request_pokemon(
        self, pokemon: Dict[str, Any], force_self_team: bool = True
    ) -> Optional[Pokemon]:
        identifier = pokemon.get("ident")
        if not identifier:
            return None
        team = self._team if force_self_team else self._opponent_team
        if identifier not in team:
            try:
                return self.get_pokemon(
                    identifier,
                    force_self_team=force_self_team,
                    details=pokemon.get("details", ""),
                    request=pokemon,
                )
            except Exception:



                team[identifier] = Pokemon(
                    request_pokemon=pokemon,
                    name=self._pokemon_name_from_ident(identifier),
                    gen=self.gen,
                )
        mon = team.get(identifier)
        if mon is not None:
            try:
                mon.update_from_request(pokemon)
            except Exception:
                pass
        return mon

    def _ingest_multi_ally_request(self, request: Dict[str, Any]) -> None:
        ally = request.get("ally")
        if not isinstance(ally, dict):
            return
        for pokemon in ally.get("pokemon", []) or []:
            mon = self._get_or_create_request_pokemon(pokemon, force_self_team=True)
            identifier = pokemon.get("ident", "")
            role = identifier[:2]
            if mon is not None and pokemon.get("active"):
                try:
                    mon.switch_in()
                except Exception:
                    pass
                self._active_pokemon[f"{role}{_multi_slot_letter(role)}"] = mon

    def _remap_own_active_slot(self) -> None:
        role = self.player_role
        if role is None:
            return
        wanted = f"{role}{_multi_slot_letter(role)}"
        wrong = f"{role}{'b' if wanted.endswith('a') else 'a'}"
        if wanted not in self._active_pokemon and wrong in self._active_pokemon:
            self._active_pokemon[wanted] = self._active_pokemon.pop(wrong)
        elif wanted in self._active_pokemon and wrong in self._active_pokemon:
            if self._active_pokemon[wrong] is self._active_pokemon[wanted] or not safe_getattr(
                self._active_pokemon[wrong], "active", True
            ):
                self._active_pokemon.pop(wrong, None)

    def _active_team_for_role(self, role: Optional[str]) -> Dict[str, Any]:
        return (
            self._active_pokemon
            if _multi_same_side(role, self.player_role)
            else self._opponent_active_pokemon
        )

    def _side_is_mine(self, player_role: str) -> bool:
        return _multi_same_side(player_role, self.player_role)


__all__ = ["MultiBattleRequestMixin"]
