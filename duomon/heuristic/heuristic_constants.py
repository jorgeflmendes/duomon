from __future__ import annotations

from typing import Dict



PROTECT_MOVES = {
    "protect",
    "detect",
    "spikyshield",
    "kingsshield",
    "banefulbunker",
    "silktrap",
    "burningbulwark",
    "maxguard",
}
HELPING_HAND_MOVES = {"helpinghand"}
BENEFICIAL_ALLY_TARGET_MOVES = {
    "helpinghand",
    "coaching",
    "healpulse",
    "floralhealing",
    "lifedew",
    "decorate",
    "psychup",
    "aromatherapy",
    "healbell",
}
SPEED_CONTROL_MOVES = {
    "tailwind",
    "icywind",
    "electroweb",
    "trickroom",
    "quash",
    "thunderwave",
    "glare",
    "stringshot",
    "bulldoze",
}
FAKE_OUT_MOVES = {"fakeout"}
REDIRECTION_MOVES = {"followme", "ragepowder"}
SLEEP_CONTROL_MOVES = {
    "spore",
    "sleeppowder",
    "hypnosis",
    "sing",
    "lovelykiss",
    "grasswhistle",
    "darkvoid",
}
STATUS_CONTROL_MOVES = SLEEP_CONTROL_MOVES | {
    "yawn",
    "taunt",
    "encore",
    "disable",
    "thunderwave",
    "glare",
    "nuzzle",
    "willowisp",
    "toxic",
}
SETUP_MOVES = {
    "swordsdance",
    "nastyplot",
    "dragondance",
    "calmmind",
    "bulkup",
    "irondefense",
    "quiverdance",
    "shellsmash",
    "coil",
    "agility",
    "rockpolish",
    "takeheart",
    "acidarmor",
    "cosmicpower",
    "amnesia",
    "autotomize",
    "shiftgear",
    "substitute",
    "bellydrum",
    "victorydance",
    "workup",
    "growth",
    "tailglow",
    "honeclaws",
    "clangoroussoul",
    "howl",
}
RECOVERY_MOVES = {
    "recover",
    "roost",
    "slackoff",
    "morningsun",
    "synthesis",
    "softboiled",
    "milkdrink",
    "shoreup",
    "strengthsap",
    "rest",
    "lifedew",
    "wish",
}
HAZARD_MOVES = {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
RELIABLE_TEMPO_SPEED_MOVES = {"icywind", "electroweb", "snarl", "strugglebug"}
PIVOT_MOVES = {"uturn", "voltswitch", "flipturn", "partingshot", "chillyreception", "shedtail"}
TERRAIN_MOVES = {
    "electricterrain",
    "grassyterrain",
    "mistyterrain",
    "psychicterrain",
    "icespinner",
    "steelroller",
}
TERRAIN_SET_MOVES = {"electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"}
TERRAIN_REMOVE_MOVES = {"icespinner", "steelroller"}
TERRAIN_BOOST_TYPES = {
    "electricterrain": "electric",
    "grassyterrain": "grass",
    "psychicterrain": "psychic",
}
SCREEN_MOVES = {"reflect", "lightscreen", "auroraveil"}
ALLY_ACTIVATION_ITEMS = {"weaknesspolicy", "absorbbulb", "cellbattery", "luminousmoss", "snowball"}
ALLY_ACTIVATION_ABILITIES = {
    "justified",
    "weakarmor",
    "stamina",
    "steamengine",
    "watercompaction",
    "motordrive",
    "voltabsorb",
    "waterabsorb",
    "flashfire",
    "sapsipper",
    "eartheater",
    "stormdrain",
    "lightningrod",
}
ALLY_MULTI_HIT_ACTIVATION_MOVES = {
    "beatup",
    "brutalswing",
    "bulldoze",
    "surf",
    "discharge",
    "lavaplume",
}
FIXED_DAMAGE_MOVES = {
    "seismictoss",
    "nightshade",
    "dragonrage",
    "sonicboom",
    "psywave",
    "finalgambit",
    "endeavor",
    "counter",
    "mirrorcoat",
    "metalburst",
}
ITEM_ACTIVATION_TYPES = {
    "absorbbulb": "water",
    "luminousmoss": "water",
    "cellbattery": "electric",
    "snowball": "ice",
}
TYPE_ABSORB_ABILITIES = {
    "dryskin": {"water"},
    "eartheater": {"ground"},
    "flashfire": {"fire"},
    "lightningrod": {"electric"},
    "motordrive": {"electric"},
    "sapsipper": {"grass"},
    "stormdrain": {"water"},
    "voltabsorb": {"electric"},
    "waterabsorb": {"water"},
}
RECOIL_MOVES = {
    "doubleedge",
    "bravebird",
    "flareblitz",
    "wavecrash",
    "headsmash",
    "woodhammer",
    "wildcharge",
    "volttackle",
    "takedown",
}
SLICING_MOVES = {
    "aquacutter",
    "airslash",
    "aerialace",
    "behemothblade",
    "bitterblade",
    "ceaselessedge",
    "crosspoison",
    "cut",
    "falseswipe",
    "furycutter",
    "kowtowcleave",
    "leafblade",
    "nightslash",
    "populationbomb",
    "psychocut",
    "razorleaf",
    "razorshell",
    "sacredsword",
    "slash",
    "solarblade",
    "stoneaxe",
    "xscissor",
}
PUNCHING_MOVES = {
    "bulletpunch",
    "cometpunch",
    "dizzypunch",
    "doubleironbash",
    "drainpunch",
    "dynamicpunch",
    "firepunch",
    "focuspunch",
    "hammerarm",
    "icehammer",
    "icepunch",
    "machpunch",
    "megapunch",
    "meteormash",
    "plasmafists",
    "poweruppunch",
    "shadowpunch",
    "skyuppercut",
    "surgingstrikes",
    "thunderpunch",
    "wickedblow",
}
SELF_SACRIFICE_MOVES = {
    "explosion",
    "selfdestruct",
    "mistyexplosion",
    "finalgambit",
    "memento",
    "healingwish",
    "lunardance",
}
SELF_DEBUFF_MOVES = {
    "closecombat",
    "leafstorm",
    "overheat",
    "dracometeor",
    "superpower",
    "makeitrain",
    "spinout",
    "psychoboost",
}
ONE_TIME_FIELD_MOVES = {
    "tailwind",
    "trickroom",
    "stealthrock",
    "spikes",
    "toxicspikes",
    "stickyweb",
    "reflect",
    "lightscreen",
    "auroraveil",
}
FIRST_TURN_ONLY_STYLE_MOVES = {"fakeout", "firstimpression"}
REPEAT_BAD_STATUS_MOVES = {
    "thunderwave",
    "glare",
    "nuzzle",
    "willowisp",
    "toxic",
    "poisongas",
    "spore",
    "sleeppowder",
    "yawn",
    "encore",
    "taunt",
    "disable",
}
LOW_PROGRESS_SUPPORT_MOVES = {
    "wish",
    "recover",
    "roost",
    "synthesis",
    "lifedew",
    "healpulse",
    "junglehealing",
    "followme",
    "ragepowder",
    "wideguard",
    "helpinghand",
    "protect",
    "detect",
    "slackoff",
    "morningsun",
    "softboiled",
    "milkdrink",
    "shoreup",
    "strengthsap",
    "rest",
    "stealthrock",
    "spikes",
    "toxicspikes",
    "stickyweb",
    "substitute",
    "acidarmor",
    "cosmicpower",
    "amnesia",
    "batonpass",
    "decorate",
    "bellydrum",
    "victorydance",
    "workup",
    "growth",
    "tailglow",
    "honeclaws",
    "clangoroussoul",
    "howl",
}
SPREAD_TARGETS = {"allAdjacent", "allAdjacentFoes", "all"}
RECHARGE_MOVES = {
    "gigaimpact",
    "hyperbeam",
    "blastburn",
    "frenzyplant",
    "hydrocannon",
    "rockwrecker",
    "roaroftime",
    "prismaticlaser",
    "eternabeam",
}





SPREAD_MOVE_IDS = {
    "alluringvoice",
    "astralbarrage",
    "bleakwindstorm",
    "blizzard",
    "boomburst",
    "bulldoze",
    "dazzlinggleam",
    "discharge",
    "earthquake",
    "eruption",
    "expandingforce",
    "glaciallance",
    "heatwave",
    "hypervoice",
    "icywind",
    "landswrath",
    "lavaplume",
    "magicaltorque",
    "makeitrain",
    "muddywater",
    "originpulse",
    "precipiceblades",
    "razorleaf",
    "rockslide",
    "sludgewave",
    "snarl",
    "strugglebug",
    "surf",
    "swift",
    "teeterdance",
    "terastarstorm",
    "thousandarrows",
    "waterspout",
    "wildboltstorm",
}
NO_EXPLICIT_TARGETS = {
    "self",
    "allAdjacentFoes",
    "allAdjacent",
    "all",
    "allies",
    "foeSide",
    "allySide",
    "field",
    "randomNormal",
    "scripted",
}




_TYPE_BOOST_ITEMS: Dict[str, set[str]] = {
    "bug": {"silverpowder"},
    "dark": {"blackglasses", "dreadplate"},
    "dragon": {"dragonfang", "dracoplate"},
    "electric": {"magnet", "zapplate"},
    "fairy": {"pixieplate"},
    "fighting": {"blackbelt", "fistplate"},
    "fire": {"charcoal", "flameplate"},
    "flying": {"sharpbeak", "skyplate"},
    "ghost": {"spelltag", "spookyplate"},
    "grass": {"miracleseed", "meadowplate", "roseincense"},
    "ground": {"softsand", "earthplate"},
    "ice": {"nevermeltice", "icicleplate"},
    "normal": {"silkscarf"},
    "poison": {"poisonbarb", "toxicplate"},
    "psychic": {"twistedspoon", "mindplate", "oddincense"},
    "rock": {"hardstone", "stoneplate", "rockincense"},
    "steel": {"metalcoat", "ironplate"},
    "water": {"mysticwater", "splashplate", "seaincense", "waveincense"},
}


__all__ = [name for name in globals() if name.isupper() or name == "_TYPE_BOOST_ITEMS"]
