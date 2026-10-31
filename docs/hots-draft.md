# Heroes of the Storm draft help

Ava reads the draft screen, so you do not have to say hero names. Everything runs on the Mac:
Apple's text recognition reads the screen, no model is loaded.

## One-time setup (macOS)

1. System Settings → Privacy & Security → **Screen & System Audio Recording** → allow the app
   that runs Ava (Terminal, iTerm or VS Code), then restart it.
2. Save the Icy Veins pages and images into `data/icyveins/raw` and `data/icyveins/images`
   (the hero portraits there read the ban slots).
3. `make run`, open <http://localhost:8080>, turn the microphone on with Ctrl+M.

## During a draft

Say "draft mode on" (or "game mode on"). Ava then watches the screen, notices a draft by herself
(the map name and a ban or pick phase on screen) and answers in a few words; speech that is
not a draft command gets "Say again." instead of going to the chat model. Outside draft mode
Ava never looks at the screen unless you ask her to read it. Say:

| You say | Ava does |
|---|---|
| "suggest", "ban", "what should I pick" | short advice from the current screen |
| "read the draft" (or "scan the screen") | map, your team, the enemy, bans, then advice for your turn |
| "read the draft and ban" | the same, then ban advice |
| "what is good against Zul'jin" | the best counters |
| "why" | the reasons of the last advice |
| "read the last screenshot" | reads the newest saved screen instead of the live one |
| "draft mode on" / "draft mode off" | switch the mode (`game_mode = true` starts every session in it) |

When the match starts Ava stays quiet until the next draft screen, and draft mode stays on.

## Names that are said badly

Names are matched by sound as well as by spelling ("Kszalatat" finds Xal'atath), limited to
heroes still available. When two heroes fit, Ava asks "One Thrall or two Tyrael?"; answer with
the number (or the name), so a name she misheard once is not needed again. The spelling is remembered in `data/hots_aliases.toml`.

## What is saved

Every screen where the draft changed is saved to `data/game_screens/<date_time>/NN.png` with a
`draft.json` of what Ava read. The last 20 drafts are kept. Nothing leaves the machine.

## Settings

`[hots.screen]` in `config/voicemate.toml`: `enabled`, `window` (app name of the game), `poll_s`, `game_mode`, `proactive` (say "Your ban." when it is your turn), `log_dir`, `keep`.
