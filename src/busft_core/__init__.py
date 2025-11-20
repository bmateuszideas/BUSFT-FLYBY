"""
Publiczny interfejs BUSFT/AM2 dla projektu BUSFT-FLYBY.

Warstwy:
- `_kernel.py`  – wewnętrzna implementacja równań (prywatna, nie tykamy).
- `space_model.py` – pomocnicze profile masy/potencjału (wewnętrzne).
- `api.py` – publiczne API, cienka warstwa delegująca do `_kernel`.

Ten moduł re-eksportuje funkcje z `api.py`, żeby reszta projektu mogła
robić po prostu: `from busft_core import baranowicz_gamma, ...`.

Formuł w `_kernel.py` NIE DOTYKAMY.
"""

from .api import *  # type: ignore  # noqa: F401,F403
