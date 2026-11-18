"""Build an 80-item question set with a difficulty spread and a type label per item.

50 multi-step arithmetic items (answers computed in code, so they are verified) and 30
harder factual items (lesser-known capitals, element numbers, history, literature).
Each item has `type` ("arithmetic" or "fact"). The answer positions are shuffled later by the
runners' `prepare_items`, as for the 50-item set.

    uv run python -m model_steering_lab.make_mixed_hard_items
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "experiments/data/mixed-hard-80.json"
SEED = 20261118

# (question, answer, three plausible wrong options)
FACTS = [
    ("What is the capital of Kyrgyzstan?", "Bishkek", ["Almaty", "Tashkent", "Dushanbe"]),
    ("What is the capital of Turkmenistan?", "Ashgabat", ["Tashkent", "Dushanbe", "Baku"]),
    ("What is the capital of Laos?", "Vientiane", ["Hanoi", "Phnom Penh", "Bangkok"]),
    ("What is the capital of Burkina Faso?", "Ouagadougou", ["Bamako", "Niamey", "Accra"]),
    ("What is the capital of Bhutan?", "Thimphu", ["Kathmandu", "Dhaka", "Gangtok"]),
    ("What is the capital of Mongolia?", "Ulaanbaatar", ["Hohhot", "Astana", "Bishkek"]),
    ("What is the capital of Suriname?", "Paramaribo", ["Georgetown", "Cayenne", "Caracas"]),
    ("What is the capital of Malawi?", "Lilongwe", ["Blantyre", "Lusaka", "Harare"]),
    ("What is the capital of Slovenia?", "Ljubljana", ["Zagreb", "Bratislava", "Sarajevo"]),
    (
        "What is the capital of Brunei?",
        "Bandar Seri Begawan",
        ["Kuala Lumpur", "Jakarta", "Manila"],
    ),
    ("What is the chemical symbol for tungsten?", "W", ["T", "Tu", "Tn"]),
    ("What is the atomic number of iron?", "26", ["24", "28", "29"]),
    ("What is the atomic number of silver?", "47", ["29", "79", "50"]),
    (
        "Which element has the highest electronegativity?",
        "Fluorine",
        ["Oxygen", "Chlorine", "Neon"],
    ),
    (
        "What is the SI unit of magnetic flux density?",
        "Tesla",
        ["Weber", "Henry", "Gauss"],
    ),
    (
        "Which particle carries the strong force between quarks?",
        "Gluon",
        ["Photon", "W boson", "Graviton"],
    ),
    (
        "Who wrote 'The Brothers Karamazov'?",
        "Fyodor Dostoevsky",
        ["Leo Tolstoy", "Ivan Turgenev", "Anton Chekhov"],
    ),
    ("In which year was the Peace of Westphalia signed?", "1648", ["1618", "1688", "1713"]),
    (
        "Which empire was ruled by Suleiman the Magnificent?",
        "Ottoman Empire",
        ["Mughal Empire", "Safavid Empire", "Byzantine Empire"],
    ),
    (
        "Who was the first Roman emperor?",
        "Augustus",
        ["Julius Caesar", "Nero", "Tiberius"],
    ),
    ("In which year was the Battle of Hastings?", "1066", ["1056", "1086", "1166"]),
    (
        "Who composed 'The Rite of Spring'?",
        "Igor Stravinsky",
        ["Sergei Prokofiev", "Claude Debussy", "Maurice Ravel"],
    ),
    (
        "Which treaty formally ended the war with Germany after World War I?",
        "Treaty of Versailles",
        ["Treaty of Trianon", "Treaty of Brest-Litovsk", "Treaty of Sevres"],
    ),
    (
        "Who wrote 'One Hundred Years of Solitude'?",
        "Gabriel Garcia Marquez",
        ["Jorge Luis Borges", "Mario Vargas Llosa", "Isabel Allende"],
    ),
    (
        "What is the longest river in Asia?",
        "Yangtze",
        ["Yellow River", "Mekong", "Ganges"],
    ),
    (
        "What is the deepest lake in the world?",
        "Lake Baikal",
        ["Lake Tanganyika", "Caspian Sea", "Lake Superior"],
    ),
    (
        "Which is the smallest bone in the human body?",
        "Stapes",
        ["Incus", "Malleus", "Hyoid"],
    ),
    (
        "Which gas makes up most of the atmosphere of Mars?",
        "Carbon dioxide",
        ["Nitrogen", "Oxygen", "Argon"],
    ),
    (
        "Which planet in our solar system has the shortest day?",
        "Jupiter",
        ["Saturn", "Mars", "Earth"],
    ),
    (
        "What is the capital of the Canadian province of Saskatchewan?",
        "Regina",
        ["Saskatoon", "Winnipeg", "Calgary"],
    ),
]


def arithmetic_items(rng: random.Random) -> list[tuple[str, int]]:
    """50 multi-step items spanning easy-to-hard for a model answering without working."""
    items: list[tuple[str, int]] = []
    for _ in range(10):
        a, b = rng.randint(21, 99), rng.randint(21, 99)
        items.append((f"What is {a} × {b}?", a * b))
    for _ in range(8):
        a, b = rng.randint(101, 499), rng.randint(11, 49)
        items.append((f"What is {a} × {b}?", a * b))
    for _ in range(8):
        a, b = rng.randint(12, 49), rng.randint(3, 9)
        c = rng.randint(50, 200)
        items.append((f"What is ({a} × {b}) + {c}?", a * b + c))
    for _ in range(8):
        a, b, c = rng.randint(30, 90), rng.randint(3, 8), rng.randint(10, 60)
        items.append((f"What is {a} × {b} − {c}?", a * b - c))
    for _ in range(8):
        pct = rng.choice([12, 15, 17, 23, 35, 45, 65, 85])
        base = rng.choice([120, 240, 340, 460, 520, 640, 780])
        value = pct * base // 100
        if pct * base % 100 == 0:
            items.append((f"What is {pct}% of {base}?", value))
    while len(items) < 42:
        pct = rng.choice([12, 15, 17, 23, 35, 45, 65, 85])
        base = rng.choice([120, 240, 340, 460, 520, 640, 780])
        if pct * base % 100 == 0:
            items.append((f"What is {pct}% of {base}?", pct * base // 100))
    for _ in range(8):
        d, q = rng.randint(12, 29), rng.randint(31, 89)
        items.append((f"What is {d * q} ÷ {d}?", q))
    return items[:50]


def distractors(answer: int, rng: random.Random) -> list[int]:
    """Three distinct, plausible wrong answers: near misses, a digit slip, a carry error."""
    pool: list[int] = []
    for delta in (10, -10, 100, -100, 1, -1, 2, -2, 20, -20):
        pool.append(answer + delta)
    text = str(answer)
    if len(text) >= 3:
        swapped = text[1] + text[0] + text[2:]
        pool.append(int(swapped))
        pool.append(int(text[:-2] + text[-1] + text[-2]))
    pool = [value for value in dict.fromkeys(pool) if value != answer and value > 0]
    rng.shuffle(pool)
    return pool[:3]


def main() -> None:
    rng = random.Random(SEED)
    items = []
    for number, (question, answer) in enumerate(arithmetic_items(rng), start=1):
        options = distractors(answer, rng)
        assert len(options) == 3 and len({answer, *options}) == 4, question
        items.append(
            {
                "id": f"hard80-{number:02d}",
                "type": "arithmetic",
                "question": question,
                "answer_text": f"{answer}",
                "distractors": [f"{value}" for value in options],
            }
        )
    for number, (question, answer, wrong) in enumerate(FACTS, start=len(items) + 1):
        assert len({answer, *wrong}) == 4, question
        items.append(
            {
                "id": f"hard80-{number:02d}",
                "type": "fact",
                "question": question,
                "answer_text": answer,
                "distractors": wrong,
            }
        )
    dataset = {
        "id": "mixed-hard-80-v1",
        "source": (
            "50 generated multi-step arithmetic items (answers computed in code) and 30 harder "
            "factual items. Built to have a difficulty spread and a type label per item."
        ),
        "choice_order_seed": SEED,
        "items": items,
    }
    OUTPUT.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {len(items)} items to {OUTPUT}")


if __name__ == "__main__":
    main()
