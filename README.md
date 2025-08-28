# Blackjack Game

Tento projekt implementuje hlavní hru Blackjack s podporou až 5 hráčů, včetně různých typů hráčů (člověk, DQN agent, Q-learning agent, Hi-Lo počítání karet). Hra je vytvořena pomocí knihovny Pygame a umožňuje interaktivní hraní s grafickým uživatelským rozhraním.

---

## Funkce

- Více hráčů (až 5)
- Herní logika Blackjacku (rozdání karet, sázení, tahy hráčů a dealera)
- Podpora AI hráčů (DQN, Q-learning, card counting)
- Grafické uživatelské rozhraní pomocí Pygame
- Ukládání a načítání hráčských dat (jméno, počet žetonů)
- Možnost vsadit, zdvojnásobit sázku, rozdělit karty a pojistit se

---

## Moduly

- Python 3.11 a vyšší
- pygame
- numpy
- random
- math
- gc
- time
- torch
- z collections defaultdict
- z sortedcontainers import SortedList
- lmdb
- os
- hashlib
- msgpack
- json
- sys

Vše lze nainstalovat pomocí
```bash
pip install -r requirements.txt
```

## Postup spuštění

Po úspěšné instalaci všech závislostí a nastavení prostředí stačí spustit hlavní skript projektu pomocí příkazu:

```bash
python main.py
```

## Jak hrát

1) Po spuštění programu se zobrazí menu, kde si můžete vybrat počet hráčů – minimálně 1 a maximálně 5. Dále si zvolíte, kdo za jednotlivé hráče bude hrát – člověk nebo agent.

2) Po nastavení hráčů stiskněte tlačítko **Start Game** pro spuštění hry.

3) Hrajte podle pravidel blackjacku.

4) Pokud zadáte sázku (bet) rovnou 0, můžete opustit hru dříve, než dosáhnete bankrotu.

5) Hru kdykoliv ukončíte pomocí tlačítka **Quit Game** v menu.

## Testování

Projekt používá framework `pytest` pro automatizované testování funkcí.

Testy spustíte příkazem:

```bash
pytest
```