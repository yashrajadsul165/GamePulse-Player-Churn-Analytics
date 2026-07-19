# GamePulse Data Dictionary

The included compressed CSV is generated with a fixed random seed and represents fictional player telemetry observed on `2026-07-01`. The target describes activity during the following 30 days.

| Column | Type | Description |
|---|---|---|
| `player_id` | string | Synthetic identifier in the form `GP-00001` |
| `observation_date` | date | Cutoff date for all input features |
| `platform` | category | Primary platform: PC, Console, or Mobile |
| `preferred_mode` | category | Casual, Ranked, Co-op, or Battle Royale |
| `region` | category | Broad fictional operating region used for dashboard slicing |
| `account_age_days` | integer | Days since the fictional account was created |
| `sessions_30d` | integer | Sessions during the 30 days before the cutoff |
| `avg_session_minutes` | decimal | Average session length during the observation period |
| `days_since_last_login` | integer | Recency at the observation cutoff |
| `matches_30d` | integer | Matches played during the prior 30 days |
| `win_rate` | decimal | Share of matches won, between 0 and 1 |
| `levels_completed_90d` | integer | Progression events during the prior 90 days |
| `social_connections` | integer | Number of fictional in-game social connections |
| `purchases_90d` | integer | Purchase events during the prior 90 days |
| `spend_usd_90d` | decimal | Fictional spend during the prior 90 days in USD |
| `battle_pass_owned` | category | Whether a fictional battle pass is owned |
| `support_tickets_90d` | integer | Support requests during the prior 90 days |
| `crashes_30d` | integer | Client crashes during the prior 30 days |
| `engagement_score` | decimal | Transparent 0–100 composite used for descriptive analytics only |
| `churned_30d` | binary | Target: 1 when the fictional player becomes inactive in the next 30 days, otherwise 0 |

## Leakage boundary

Only columns observed on or before the cutoff are model inputs. `engagement_score`, `region`, identifiers, and the future target are not model features. The model feature list is defined once in `src/data.py` and reused by training and scoring code.

## Privacy

Every row is synthetic. Names, email addresses, device IDs, IP addresses, chat messages, precise locations, and biometric data are intentionally absent.
