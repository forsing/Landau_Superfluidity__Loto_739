#!/usr/bin/env python3

"""Landau V3: duge i lokalne pobude, energija i izbor cele kombinacije."""

import argparse
import csv
from pathlib import Path
import numpy as np

DEFAULTS = (
    "/Users/4c/Desktop/GHQ/data/loto7_4698_k80.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_2971.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_plus_1727.csv",
)

N, K, WINDOW = 39, 7, 64
SETTINGS = tuple(
    (a, b)
    for a in (.01, .1, 1.)
    for b in (.01, .1, 1.)
)


def load(path):
    rows = []

    with Path(path).open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if not row or all(not v.strip() for v in row):
                continue

            try:
                values = [int(v.strip()) for v in row]
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: ocekujem cele brojeve."
                ) from error

            if (
                len(values) != K
                or values != sorted(values)
                or len(set(values)) != K
                or not all(1 <= v <= N for v in values)
            ):
                raise ValueError(
                    f"Red {line}: neispravna sortirana "
                    "7/39 kombinacija."
                )

            rows.append(values)

    return np.asarray(rows, dtype=int).reshape(-1, K)


def haar():
    basis = [np.ones(8)/np.sqrt(8)]

    for width in (8, 4, 2):
        for start in range(0, 8, width):
            vector = np.zeros(8)
            vector[start:start+width//2] = 1/np.sqrt(width)
            vector[start+width//2:start+width] = -1/np.sqrt(width)
            basis.append(vector)

    return np.asarray(basis)


H = haar()
MODES = H[1:]


def gaps(rows):
    rows = np.atleast_2d(rows)

    edges = np.column_stack((
        np.zeros(len(rows)),
        rows,
        np.full(len(rows), 40)
    ))

    return np.diff(edges, axis=1)-1.


def excitations(rows, reference):
    # Referenca je prethodna STVARNA kombinacija.
    # Nema prosecne pozadine.
    return (
        (gaps(rows)-gaps(reference)) @ MODES.T / 5.
    )


def replacements(row):
    unused = np.setdiff1d(np.arange(1, 40), row)
    result = np.tile(row, (K*len(unused), 1))

    for j in range(K):
        result[j*len(unused):(j+1)*len(unused), j] = unused

    return np.sort(result, axis=1)


def design(rows, reference, previous):
    e = excitations(rows, reference)

    # Potpisane pobude, linearna/kvadratna disperzija,
    # sprega dugih i lokalnih kanala i prethodni tok.
    coupling = (
        e[:, :3, None] * e[:, None, 3:]
    ).reshape(len(e), 12)

    flow = (
        e[:, :, None] * previous[None, None, :]
    ).reshape(len(e), 49)

    return np.column_stack((
        e, np.abs(e), e*e, coupling, flow
    ))


def penalties(alpha_long, alpha_local):
    mode = np.r_[
        np.full(3, alpha_long),
        np.full(4, alpha_local)
    ]

    return np.r_[
        mode,
        mode,
        mode,
        np.full(12, np.sqrt(alpha_long*alpha_local)),
        np.repeat(mode, 7)
    ]


def prepare(history):
    if len(history) < WINDOW+2:
        raise ValueError(
            "Potrebno je najmanje 66 prethodnih redova."
        )

    history = history[-(WINDOW+2):]
    gram = np.zeros((82, 82))
    target = np.zeros(82)

    # Ucenje razlike skorova stvarnog naslednika
    # i svih njegovih zamena jednog broja.
    for t in range(2, len(history)):
        previous = excitations(
            history[t-1], history[t-2]
        )[0]

        actual = design(
            history[t:t+1], history[t-1], previous
        )
        alternatives = design(
            replacements(history[t]),
            history[t-1],
            previous
        )

        contrasts = actual-alternatives

        gram += (
            contrasts.T @ contrasts / len(contrasts)
        )
        target += contrasts.sum(axis=0) / len(contrasts)

    gram /= WINDOW
    target /= WINDOW

    candidates = np.unique(
        np.vstack(
            [history[2:]]
            + [
                replacements(row)
                for row in history[2:]
            ]
        ),
        axis=0
    )

    previous = excitations(
        history[-1], history[-2]
    )[0]

    features = design(
        candidates, history[-1], previous
    )

    return (
        gram, target, candidates, features,
        history[-1], previous
    )


def fit(prepared, alpha_long, alpha_local):
    gram, target = prepared[:2]

    return np.linalg.solve(
        gram + np.diag(
            penalties(alpha_long, alpha_local)
        ),
        target
    )


def choose(prepared, coefficient):
    candidates, features = prepared[2:4]
    scores = features @ coefficient

    if not np.all(np.isfinite(scores)):
        raise ArithmeticError(
            "Neispravna energija pobuda."
        )

    best = int(np.argmax(scores))

    return (
        tuple(map(int, candidates[best])),
        float(scores[best])
    )


def predict(prepared, alpha_long, alpha_local):
    return choose(
        prepared,
        fit(prepared, alpha_long, alpha_local)
    )


def energy(combo, reference, previous, coefficient):
    return -float(
        design(combo, reference, previous)[0] @ coefficient
    )


def hits(prediction, actual):
    return len(
        set(map(int, prediction))
        & set(map(int, actual))
    )


def summarize(label, scores):
    print(
        f"{label}: prosek={np.mean(scores):.4f}; "
        f"pogodaka 0..7="
        f"{np.bincount(scores, minlength=8).tolist()}",
        flush=True
    )


def run_file(path, validation_count=48, test_count=64):
    rows = load(path)
    split = len(rows)-test_count
    begin = split-validation_count

    if (
        validation_count < 16
        or test_count < 16
        or begin < WINDOW+2
    ):
        raise ValueError(
            "Premalo istorije ili manje od "
            "16 validacionih/test kola."
        )

    print(
        f"\nUlaz: {Path(path).resolve()}; "
        f"redova={len(rows)}",
        flush=True
    )

    validation = [[] for _ in SETTINGS]

    for t in range(begin, split):
        prepared = prepare(rows[:t])

        for j, setting in enumerate(SETTINGS):
            combo, _ = predict(prepared, *setting)
            validation[j].append(hits(combo, rows[t]))

        if (t-begin+1) % 16 == 0:
            print(
                f"Validacija: "
                f"{t-begin+1}/{validation_count}",
                flush=True
            )

    means = [np.mean(s) for s in validation]
    chosen = int(np.argmax(means))
    setting = SETTINGS[chosen]

    print(
        f"Validacija: redovi {begin+1}..{split}; "
        f"alpha_long={setting[0]:g}; "
        f"alpha_local={setting[1]:g}; "
        f"prosek={means[chosen]:.4f}",
        flush=True
    )

    scores, previous_scores = [], []

    # Regularizacija je izabrana pre testa.
    # Svako kolo koristi samo prethodne redove.
    for t in range(split, len(rows)):
        combo, _ = predict(
            prepare(rows[:t]), *setting
        )

        scores.append(hits(combo, rows[t]))
        previous_scores.append(
            hits(rows[t-1], rows[t])
        )

        if (t-split+1) % 16 == 0:
            print(
                f"Test: {t-split+1}/{test_count}",
                flush=True
            )

    print(f"Test: redovi {split+1}..{len(rows)}")
    summarize("LANDAU V3 SUPERFLUIDITY", scores)
    summarize(
        "Prethodna kombinacija", previous_scores
    )

    prepared = prepare(rows)
    combo, score = predict(prepared, *setting)
    e = excitations(combo, rows[-1])[0]

    print(
        f"Poslednji red: "
        f"{','.join(map(str, rows[-1]))}"
    )
    print(
        f"SLEDECI RED {len(rows)+1}: "
        f"{','.join(map(str, combo))}"
    )
    print(
        f"Skor S=-E: {score:.10f}; "
        f"kandidata={len(prepared[2])}"
    )
    print(
        f"Jacina dugih pobuda="
        f"{np.linalg.norm(e[:3]):.6f}; "
        f"lokalnih={np.linalg.norm(e[3:]):.6f}"
    )
    print(
        "Izbor: poslednje 64 kombinacije "
        "i sve njihove zamene jednog broja."
    )
    print(
        "Pobude su strukturna analogija; "
        "skor nije verovatnoca."
    )


def self_test():
    assert np.allclose(H @ H.T, np.eye(8))
    assert np.allclose(MODES.sum(axis=1), 0.)

    base = np.array([
        [1,2,3,4,5,6,7],
        [4,12,18,28,35,36,37],
        [33,34,35,36,37,38,39],
        [2,7,12,18,24,30,39]
    ])

    for reference in base:
        e = excitations(base, reference)
        restored = gaps(reference) + 5.*e @ MODES

        assert np.allclose(restored, gaps(base))
        assert np.allclose(
            np.cumsum(
                restored[:, :-1]+1, axis=1
            ),
            base
        )
        assert np.allclose(
            np.sum(
                (gaps(base)-gaps(reference))**2,
                axis=1
            ),
            25.*np.sum(e*e, axis=1)
        )

        neighbors = replacements(reference)

        assert (
            len(neighbors)
            == len(np.unique(neighbors, axis=0))
            == 224
        )
        assert all(
            hits(reference, x) == 6
            for x in neighbors
        )

    history = np.tile(base, (20, 1))
    prepared = prepare(history)

    assert (
        np.linalg.eigvalsh(prepared[0]).min()
        >= -1e-8
    )

    for setting in SETTINGS:
        coef = fit(prepared, *setting)
        result = choose(prepared, coef)

        assert result == predict(
            prepare(history), *setting
        )
        assert any(
            np.array_equal(result[0], x)
            for x in prepared[2]
        )
        assert np.isclose(
            result[1],
            -energy(
                result[0],
                prepared[4],
                prepared[5],
                coef
            )
        )

    altered = history.copy()
    altered[-1] = base[0]

    assert (
        predict(prepare(history[:-1]), .1, .1)
        == predict(prepare(altered[:-1]), .1, .1)
    )

    print(
        "Self-test OK: modovi, rekonstrukcija, "
        "norma, rangiranje i hronologija."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="*")
    parser.add_argument("--validation", type=int, default=48)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    for path in args.csv or DEFAULTS:
        run_file(path, args.validation, args.test)


if __name__ == "__main__":
    main()



"""
Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80.csv; redova=4698
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 4587..4634; alpha_long=0.1; alpha_local=1; prosek=1.3958
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 4635..4698
LANDAU V3 SUPERFLUIDITY: prosek=1.2188; pogodaka 0..7=[15, 26, 18, 4, 1, 0, 0, 0]
Prethodna kombinacija: prosek=1.2188; pogodaka 0..7=[13, 30, 16, 4, 1, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 4699: 4,5,6,11,12,28,29
Skor S=-E: 1.2990448930; kandidata=14396
Jacina dugih pobuda=2.424871; lokalnih=3.328663
Izbor: poslednje 64 kombinacije i sve njihove zamene jednog broja.
Pobude su strukturna analogija; skor nije verovatnoca.



Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_2971.csv; redova=2971
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 2860..2907; alpha_long=0.01; alpha_local=1; prosek=1.2292
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 2908..2971
LANDAU V3 SUPERFLUIDITY: prosek=1.2656; pogodaka 0..7=[13, 24, 24, 3, 0, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[11, 30, 16, 7, 0, 0, 0, 0]
Poslednji red: 18,20,21,24,29,34,38
SLEDECI RED 2972: 1,12,13,14,15,20,22
Skor S=-E: 2.2721874326; kandidata=14392
Jacina dugih pobuda=2.366432; lokalnih=4.516636
Izbor: poslednje 64 kombinacije i sve njihove zamene jednog broja.
Pobude su strukturna analogija; skor nije verovatnoca.



Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_plus_1727.csv; redova=1727
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 1616..1663; alpha_long=1; alpha_local=0.01; prosek=1.3542
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 1664..1727
LANDAU V3 SUPERFLUIDITY: prosek=1.1875; pogodaka 0..7=[9, 39, 12, 3, 1, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[15, 23, 18, 8, 0, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 1728: 9,10,29,30,31,34,37
Skor S=-E: 2.0070585246; kandidata=14388
Jacina dugih pobuda=0.894427; lokalnih=3.730952
Izbor: poslednje 64 kombinacije i sve njihove zamene jednog broja.
Pobude su strukturna analogija; skor nije verovatnoca.
"""



"""
Pobude se računaju prema prethodnoj stvarnoj kombinaciji. 
Model uči njihovu energiju i spregu, pa bira celu kombinaciju bez prosečenja položaja ili amplituda.

Fajl	                Sledeća kombinacija	Prosek pogodaka na 64 kola	Prethodna kombinacija
Objedinjeni, red 4699	4,5,6,11,12,28,29	1,2188	                    1,2188
Loto, red 2972	        1,12,13,14,15,20,22	1,2656	                    1,2969
Loto Plus, red 1728	    9,10,29,30,31,34,37	1,1875	                    1,2969

Na ovom testu nije pokazana prednost nad prethodnom kombinacijom. 
Provereni su rekonstrukcija, ponovljivost i hronologija. 
Pretraga obuhvata poslednje 64 kombinacije i sve njihove zamene jednog broja.
"""
