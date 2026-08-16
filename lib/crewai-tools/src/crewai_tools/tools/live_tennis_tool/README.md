# Live Tennis Tool

Query professional tennis data from the [Live Tennis API](https://livetennisapi.com) — live match scores, upcoming matches, scheduled fixtures, player search and profiles, ranking tables, and API usage.

The tool is scoped to REST endpoints, most of which are available on the free tier, so it can be tried without payment.

## Actions

| `action` | Endpoint | Plan | Notes |
|---|---|---|---|
| `live_matches` | `GET /matches?status=live` | Free | Matches in play with current scores; optional `tour` |
| `upcoming_matches` | `GET /matches?status=upcoming` | Free | Matches starting soon; optional `tour` |
| `fixtures` | `GET /fixtures` | Free | Scheduled matches; optional `tour` |
| `search_players` | `GET /players?search=` | Free | Requires `search` |
| `player_profile` | `GET /players/{id}` | Free | Requires `player_id` (numeric id from `search_players`); includes current ranking |
| `rankings` | `GET /rankings?system=` | PRO | Requires `system` (`atp`, `wta`, `itf_jt`, `itf_mt`, `itf_wt`); UTR has no listing (it is a rating, not a ranking) |
| `usage` | `GET /usage` | Free | Your quota vs. consumption; exempt from quota |

### Parameters

- `tour` — optional filter for `live_matches`, `upcoming_matches` and `fixtures`: one of `atp`, `wta`, `challenger`, `itf`, `juniors`. Omit for all tours.
- `limit` / `offset` — pagination for the list actions (`live_matches`, `upcoming_matches`, `fixtures`, `search_players`, `rankings`). The API default is 50 results.

The free tier is keyed and rate-limited to 30 requests/minute and 100 requests/day. Completed-match history is a paid feature and is not part of this tool. The API also offers a WebSocket push feed and model win probability on its top tier; this tool intentionally sticks to the polling REST surface.

## Environment Variables

```env
LIVETENNIS_API_KEY=your_api_key
```

Get a free key at [livetennisapi.com](https://livetennisapi.com). Full API documentation: [docs.livetennisapi.com](https://docs.livetennisapi.com).

## Example Usage

```python
import json

from crewai_tools import LiveTennisTool

tool = LiveTennisTool()

# Matches in play right now
print(tool.run(action="live_matches", tour="atp"))

# Find a player, then load their profile using the id from the search result
players = json.loads(tool.run(action="search_players", search="alcaraz"))
player_id = players["data"][0]["id"]
profile = tool.run(action="player_profile", player_id=player_id)
```

## With an Agent

```python
from crewai import Agent
from crewai_tools import LiveTennisTool

reporter = Agent(
    role="Tennis Reporter",
    goal="Summarise what is happening on tour right now",
    backstory="You follow professional tennis and report live developments.",
    tools=[LiveTennisTool()],
)
```

## Error Handling

The tool returns readable messages (rather than raising) when the API key is missing, invalid (401), the endpoint needs a higher plan (403), or the rate limit is hit (429), so an agent can recover or report the problem.
