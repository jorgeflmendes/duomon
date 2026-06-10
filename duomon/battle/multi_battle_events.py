from __future__ import annotations

from .multi_battle_context import *


class MultiBattleEventMixin:
    def parse_message(self, split_message: List[str]):
        self._record_short_memory_reveal_event(split_message)
        if len(split_message) <= 1 or split_message[1] != "move":
            if len(split_message) > 2 and split_message[1] == "error":
                self._last_choice_error_rqid = (self._last_raw_request or {}).get("rqid")
                logger.warning(
                    f"{self.battle_tag} | Showdown rejeitou escolha: {' | '.join(map(str, split_message[2:]))}"
                )
            try:
                return super().parse_message(split_message)
            except KeyError:




                if len(split_message) > 1 and split_message[1] in {"-sideend", "-fieldend"}:
                    return
                raise

        self._replay_data.append(split_message[:])
        event = split_message[:]
        try:
            pokemon = event[2]
            mon = self.get_pokemon(pokemon)
            use = not safe_getattr(mon, "_dancing", False)
            failed = False
            reveal = not safe_getattr(mon, "_dancing", False)
            overridden_move = None
            spread = False
            try:
                mon._dancing = False
            except Exception:
                pass

            for suffix in ["[miss]", "[still]", "[notarget]"]:
                if event and event[-1] == suffix:
                    event = event[:-1]
                    failed = True
            if event and event[-1] == "[notarget]":
                event = event[:-1]
            while event and str(event[-1]).startswith("[spread]"):
                spread = True
                event = event[:-1]
            if event and event[-1] in {
                "[from] lockedmove",
                "[from]lockedmove",
                "[from] Sky Attack",
            }:
                use = False
                reveal = False
                event = event[:-1]
            if event and event[-1] in {"[from] Pursuit", "[from]Pursuit", "[zeffect]"}:
                event = event[:-1]
            if event and event[-1] == "[from] Sleep Talk":
                event[-1] = "[from] move: Sleep Talk"
            if event and str(event[-1]).startswith("[anim]"):
                event = event[:-1]
            if event and str(event[-1]).startswith(("[from] move: ", "[from]move: ")):
                overridden_move = event.pop().split(": ")[-1]
                if overridden_move in {"Copycat", "Metronome", "Nature Power", "Round"}:
                    reveal = False
                elif overridden_move in {"Grass Pledge", "Water Pledge", "Fire Pledge"}:
                    overridden_move = None
            if event and event[-1] == "null":
                event = event[:-1]
            if event and str(event[-1]).startswith(("[from] ability: ", "[from]ability: ")):
                revealed_ability = event.pop().split(": ")[-1]
                self.get_pokemon(event[2]).ability = revealed_ability
                if revealed_ability == "Magic Bounce":
                    use = False
                    reveal = False
                elif revealed_ability == "Dancer":
                    return
            if event and event[-1] in {"[from] Magic Coat", "[from] Mirror Move"}:
                use = False
                reveal = False
                event = event[:-1]
            while event and event[-1] == "[still]":
                event = event[:-1]

            presumed_target = None
            if len(event) == 4:
                pokemon, move = event[2:4]
            elif len(event) >= 5:
                pokemon, move, presumed_target = event[2:5]
                if presumed_target == "":
                    presumed_target = None
            else:
                return



            if presumed_target is not None:
                target_head = presumed_target[:4]
                if target_head not in VALID_MULTI_TARGET_HEADS:
                    presumed_target = None

            if str(move).upper().strip() == "MINIMIZE":
                self.get_pokemon(pokemon).start_effect("MINIMIZE")
            if spread:
                presumed_target = None
            self._record_short_memory_move_event(pokemon, move, presumed_target, failed)

            try:
                pressure = self._pressure_on(pokemon, move, presumed_target)
            except Exception:
                pressure = 0
            mon = self.get_pokemon(pokemon)
            try:
                role = self._normalize_multi_ident(pokemon)[:2]
                active_key = f"{role}{_multi_slot_letter(role)}"
                active_team = self._active_team_for_role(role)
                if mon is not None and not bool(safe_getattr(mon, "fainted", False)):
                    mon.switch_in()
                    active_team[active_key] = mon
            except Exception:
                pass
            if overridden_move:
                mon.moved(move, failed=failed, use=False, reveal=reveal)
                try:
                    overridden = mon.moves[Move.retrieve_id(overridden_move)]
                    overridden.use(pressure, overridden=True)
                except Exception:
                    pass
            elif not failed and move in {"Sleep Talk", "Copycat", "Metronome", "Nature Power"}:
                mon.moved(move, failed=failed, use=use, reveal=reveal)
            else:
                mon.moved(move, failed=failed, use=use, reveal=reveal, pressure=pressure)
        except Exception:


            try:
                self.get_pokemon(split_message[2]).moved(
                    split_message[3], failed=False, use=True, reveal=True
                )
            except Exception:
                return

    def _record_short_memory_reveal_event(self, split_message: List[str]) -> None:
        if len(split_message) < 4:
            return
        event = str(split_message[1] or "")
        if event not in {"-item", "-enditem", "-ability", "switch", "drag", "replace"}:
            return
        try:
            ident = self._normalize_multi_ident(str(split_message[2] or ""))
            role = ident[:2]
            same_side = _multi_same_side(role, self.player_role)
            mon = self.get_pokemon(ident)
            species = safe_species(mon)
            mem = _get_multi_short_memory(self, self.player_role)
            bucket_name = "ally_facts" if same_side else "enemy_facts"
            facts = mem.setdefault(bucket_name, {}).setdefault(species, {"species": species})
            facts["role"] = role
            facts["last_seen_turn"] = int(safe_getattr(self, "turn", 0) or 0)
            if event == "-item":
                facts["item"] = safe_compact_name(split_message[3])
                facts["item_source"] = "showdown_event"
            elif event == "-enditem":
                facts["consumed_item"] = safe_compact_name(split_message[3])
                facts["item"] = "none"
                facts["item_source"] = "showdown_event"
            elif event == "-ability":
                facts["ability"] = safe_compact_name(split_message[3])
                facts["ability_source"] = "showdown_event"
            else:
                facts["hp"] = round(float(safe_hp_fraction(mon)), 4)
                facts["types"] = _mon_type_names(mon)
        except Exception:
            return

    def _record_short_memory_move_event(
        self, pokemon: str, move: str, presumed_target: Optional[str], failed: bool
    ) -> None:
        try:
            ident = self._normalize_multi_ident(str(pokemon or ""))
            role = ident[:2]
            same_side = _multi_same_side(role, self.player_role)
            mon = self.get_pokemon(ident)
            species = safe_species(mon)
            mem = _get_multi_short_memory(self, self.player_role)
            bucket_name = "ally_facts" if same_side else "enemy_facts"
            facts = mem.setdefault(bucket_name, {}).setdefault(species, {"species": species})
            turn = int(safe_getattr(self, "turn", 0) or 0)
            mid = str(move or "").replace(" ", "").replace("-", "").lower()
            facts["role"] = role
            facts["last_seen_turn"] = turn
            facts.setdefault("revealed_moves", {})[mid] = {
                "name": str(move or ""),
                "last_turn": turn,
                "failed": bool(failed),
            }
            if presumed_target:
                target_ident = self._normalize_multi_ident(str(presumed_target or ""))
                facts["last_target_role"] = target_ident[:2]
                facts["last_target"] = self._pokemon_name_from_ident(target_ident)
            if not same_side:
                mem.setdefault("enemy_events", []).append(
                    {
                        "turn": turn,
                        "species": species,
                        "role": role,
                        "move_id": mid,
                        "target_role": facts.get("last_target_role"),
                        "failed": bool(failed),
                    }
                )
                mem["enemy_events"] = mem["enemy_events"][-16:]
        except Exception:
            return


__all__ = ["MultiBattleEventMixin"]
