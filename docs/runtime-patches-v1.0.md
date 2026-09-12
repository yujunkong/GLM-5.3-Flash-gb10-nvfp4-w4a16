# Runtime patches inventory (v1.0.0-golden)

Canonical decisions: [`golden-configuration.md`](golden-configuration.md) · [`performance.md`](performance.md).

| ID | Patch | In repo | Golden | Evidence |
|----|-------|---------|--------|----------|
| 01 | Spin-wait 0.002 | `runtime/spinwait/*.py` | **pristine** | TPS −1.63%, drift worse |
| 02 | expandable_segments | compose / `.env` | **True** | OFF −3.99% |
| 03 | Logits budget | compose / `.env` | **512** | 64 → −2.77% |
| 04 | TPS drift logger | `scripts/tps_drift_logger.py` | optional ops | no decode impact |
| 05 | Top-K GB10 fallback | image + kpool overlay | already on | PATCH=NONE |

Do **not** enable 01 or logits=64 on production. Drift improvement on this stack came from **2400 clock lock**, not spin-wait.
