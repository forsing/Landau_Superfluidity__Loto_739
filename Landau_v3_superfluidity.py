#!/usr/bin/env python3

"""LANDAU V3 — SUPERFLUIDITY OF HELIUM II: coupled structural excitations."""

import argparse
import csv
import itertools
from pathlib import Path
import numpy as np



DEFAULT = "/Users/4c/Desktop/GHQ/data/loto7_4698_k80.csv"
# DEFAULT = "/Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_2971.csv"
# DEFAULT = "/Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_plus_1727.csv"



N, K, START, WINDOW, BACKGROUND = 39, 7, 32, 768, 32


def load(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        for line, row in enumerate(csv.reader(f), 1):
            if not row or all(not v.strip() for v in row):
                continue
            try:
                numbers = [int(v.strip()) for v in row]
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: ocekujem sedam celih brojeva."
                ) from error

            if (len(numbers) != K or numbers != sorted(numbers)
                    or len(set(numbers)) != K
                    or not all(1 <= v <= N for v in numbers)):
                raise ValueError(
                    f"Red {line}: neispravna sortirana 7/39 kombinacija."
                )
            rows.append(numbers)

    return np.asarray(rows, dtype=float).reshape(-1, K)


def haar():
    size = K + 1
    basis = [np.ones(size) / np.sqrt(size)]
    width = size

    while width >= 2:
        for start in range(0, size, width):
            vector = np.zeros(size)
            vector[start:start+width//2] = 1 / np.sqrt(width)
            vector[start+width//2:start+width] = -1 / np.sqrt(width)
            basis.append(vector)
        width //= 2

    return np.asarray(basis)


H = haar()
MODES = H[1:]


def gaps(x, n=N):
    return np.diff(np.r_[0., x, float(n+1)]) - 1.


def excitation_data(rows):
    g = np.asarray([gaps(row) for row in rows])
    cumulative = np.vstack((
        np.zeros(K+1),
        np.cumsum(g, axis=0),
    ))

    background = np.empty((len(rows)+1, K+1))
    background[0] = (N-K) / (K+1)

    # Pozadina za ciljni red koristi samo prethodne redove.
    for t in range(1, len(rows)+1):
        begin = max(0, t-BACKGROUND)
        background[t] = (
            cumulative[t]-cumulative[begin]
        ) / (t-begin)

    amplitudes = (g-background[:-1]) @ MODES.T
    features = []

    for t in range(START, len(rows)+1):
        previous = amplitudes[t-1]

        # Sprega dugih i lokalnih pobuda.
        coupling = np.outer(
            previous[:3], previous[3:]
        ).ravel()

        features.append(np.r_[
            previous,
            amplitudes[t-2],
            amplitudes[t-8:t].mean(axis=0),
            amplitudes[t-32:t, :3].mean(axis=0),
            coupling,
            previous[3:]**2,
        ])

    return background, amplitudes, np.asarray(features)


def prepare(data, t):
    background, amplitudes, features = data
    begin = max(START, t-WINDOW)

    x = features[begin-START:t-START]
    center = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-8)

    a = np.column_stack((
        np.ones(len(x)),
        (x-center) / scale,
    ))
    y = amplitudes[begin:t]
    query = np.r_[
        1., (features[t-START]-center) / scale
    ]

    return (
        a,
        y,
        query,
        a.T @ a,
        a.T @ y,
        background[t],
    )


def fit(prepared, alpha_long, alpha_local):
    a, y, query, gram, rhs, background = prepared
    penalty = np.diag(np.r_[
        0., np.ones(a.shape[1]-1)
    ])

    long_coef = np.linalg.solve(
        gram + alpha_long*penalty,
        rhs[:, :3],
    )
    local_coef = np.linalg.solve(
        gram + alpha_local*penalty,
        rhs[:, 3:],
    )
    coefficients = np.column_stack((
        long_coef, local_coef
    ))

    predicted_amplitudes = query @ coefficients
    target_gaps = (
        background + predicted_amplitudes @ MODES
    )
    target = np.cumsum(target_gaps[:-1]+1)

    residual_gaps = (y-a @ coefficients) @ MODES
    residual_positions = np.cumsum(
        residual_gaps[:, :-1], axis=1
    )

    ps = np.maximum(
        1., np.median(np.abs(residual_positions), axis=0)
    )
    gs = np.maximum(
        1., np.median(np.abs(residual_gaps), axis=0)
    )

    return (
        target,
        ps,
        gs,
        predicted_amplitudes,
        target_gaps,
    )


def cost(combo, target, ps, gs, n=N):
    return float(
        (np.abs(np.asarray(combo)-target)/ps).sum()
        + (
            np.abs(gaps(combo, n)-gaps(target, n))/gs
        ).sum()
    )


def decode(target, ps, gs, n=N):
    # Tacan globalni minimum nad svim rastucim kombinacijama.
    k = len(target)
    target_gaps = gaps(target, n)
    table = np.full((k, n+1), np.inf)
    parent = np.full((k, n+1), -1, dtype=int)

    for v in range(1, n-k+2):
        table[0, v] = (
            abs(v-target[0])/ps[0]
            + abs(v-1-target_gaps[0])/gs[0]
        )

    for j in range(1, k):
        for v in range(j+1, n-k+j+2):
            previous = np.arange(j, v)
            choices = (
                table[j-1, previous]
                + np.abs(
                    v-previous-1-target_gaps[j]
                )/gs[j]
            )
            best = int(np.argmin(choices))
            parent[j, v] = previous[best]
            table[j, v] = (
                choices[best] + abs(v-target[j])/ps[j]
            )

    ends = np.arange(k, n+1)
    scores = (
        table[k-1, ends]
        + np.abs(n-ends-target_gaps[k])/gs[k]
    )
    best = int(np.argmin(scores))
    v = int(ends[best])
    combo = [v]

    for j in range(k-1, 0, -1):
        v = int(parent[j, v])
        combo.append(v)

    return tuple(reversed(combo)), float(scores[best])


def hits(prediction, actual):
    return len(
        set(map(int, prediction)) & set(map(int, actual))
    )


def summarize(label, scores):
    print(
        f"{label}: prosek={np.mean(scores):.4f}; "
        f"pogodaka 0..7="
        f"{np.bincount(scores, minlength=8).tolist()}"
    )


def self_test():
    assert np.allclose(H @ H.T, np.eye(K+1))
    assert np.allclose(MODES.sum(axis=1), 0)

    rows = np.tile(np.array([
        [1, 2, 3, 4, 5, 6, 7],
        [4, 12, 18, 28, 35, 36, 37],
        [33, 34, 35, 36, 37, 38, 39],
        [2, 7, 12, 18, 24, 30, 39],
    ], dtype=float), (50, 1))

    data = excitation_data(rows)
    original_gaps = np.asarray([
        gaps(row) for row in rows
    ])

    assert np.allclose(
        data[0][:-1] + data[1] @ MODES,
        original_gaps,
    )

    t = 170
    result = fit(prepare(data, t), 10., 100.)

    changed = rows.copy()
    changed[t:] = [3, 8, 13, 19, 25, 31, 38]
    altered = fit(
        prepare(excitation_data(changed), t),
        10.,
        100.,
    )

    for a, b in zip(result, altered):
        assert np.array_equal(a, b)

    target, ps, gs, amplitudes, predicted_gaps = result
    assert np.isclose(predicted_gaps.sum(), N-K)
    assert np.allclose(gaps(target), predicted_gaps)
    assert decode(target, ps, gs) == decode(target, ps, gs)

    target = np.array([8., 2., 5.])
    ps = np.array([1., 1.4, 1.1])
    gs = np.array([1., 1.3, 1.2, 1.])

    combo, score = decode(target, ps, gs, n=9)
    exact = min(
        cost(c, target, ps, gs, n=9)
        for c in itertools.combinations(range(1, 10), 3)
    )

    assert np.isclose(
        score, exact, rtol=1e-12, atol=1e-12
    )
    assert np.isclose(
        score, cost(combo, target, ps, gs, n=9)
    )

    print(
        "Self-test OK: rekonstrukcija, ocuvanje raspona, "
        "hronologija i globalni minimum."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default=DEFAULT)
    parser.add_argument("--validation", type=int, default=48)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    rows = load(args.csv)
    if args.validation < 16 or args.test < 16:
        parser.error(
            "Validation i test moraju imati bar 16 kola."
        )

    split = len(rows)-args.test
    begin = split-args.validation
    if begin < START+128:
        parser.error(
            "Premalo istorije za zadatu validaciju i test."
        )

    data = excitation_data(rows)
    configurations = tuple(itertools.product(
        (1., 10., 100.), repeat=2
    ))
    validation = [[] for _ in configurations]

    print(
        f"Ulaz: {Path(args.csv).resolve()}; "
        f"redova={len(rows)}",
        flush=True,
    )

    for t in range(begin, split):
        prepared = prepare(data, t)

        for j, (alpha_long, alpha_local) in enumerate(
            configurations
        ):
            target, ps, gs, _, _ = fit(
                prepared, alpha_long, alpha_local
            )
            combo, _ = decode(target, ps, gs)
            validation[j].append(hits(combo, rows[t]))

    means = [
        np.mean(values) for values in validation
    ]
    best = int(np.argmax(means))
    alpha_long, alpha_local = configurations[best]

    print(
        f"Validacija: redovi {begin+1}..{split}; "
        f"alpha_long={alpha_long:g}; "
        f"alpha_local={alpha_local:g}; "
        f"prosek={means[best]:.4f}",
        flush=True,
    )

    model_scores = []
    previous_scores = []
    background_scores = []

    for t in range(split, len(rows)):
        target, ps, gs, _, _ = fit(
            prepare(data, t),
            alpha_long,
            alpha_local,
        )
        combo, _ = decode(target, ps, gs)

        model_scores.append(hits(combo, rows[t]))
        previous_scores.append(hits(rows[t-1], rows[t]))

        reference_target = np.cumsum(
            data[0][t, :-1]+1
        )
        reference, _ = decode(
            reference_target,
            np.ones(K),
            np.ones(K+1),
        )
        background_scores.append(hits(reference, rows[t]))

    print(f"Test: redovi {split+1}..{len(rows)}")
    summarize("LANDAU V3 SUPERFLUIDITY HE II", model_scores)
    summarize("Prethodna kombinacija", previous_scores)
    summarize("Pozadina bez pobuda", background_scores)

    target, ps, gs, amplitudes, predicted_gaps = fit(
        prepare(data, len(rows)),
        alpha_long,
        alpha_local,
    )
    combo, score = decode(target, ps, gs)

    print(
        f"Poslednji red: "
        f"{','.join(map(str, rows[-1].astype(int)))}"
    )
    print(
        f"SLEDECI RED {len(rows)+1}: "
        f"{','.join(map(str, combo))}"
    )
    print(
        f"Trosak projekcije={score:.10f}; "
        "globalni minimum potvrden."
    )
    print(
        f"Jacina dugih pobuda="
        f"{np.linalg.norm(amplitudes[:3]):.6f}; "
        f"lokalnih pobuda="
        f"{np.linalg.norm(amplitudes[3:]):.6f}"
    )
    print(
        "Kanali pobuda su strukturna analogija; "
        "trosak rangiranja nije verovatnoca."
    )


if __name__ == "__main__":
    main()



"""
Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80.csv; redova=4698
Validacija: redovi 4587..4634; alpha_long=1; alpha_local=1; prosek=1.3333
Test: redovi 4635..4698
LANDAU V3 SUPERFLUIDITY HE II: prosek=1.3594; pogodaka 0..7=[11, 30, 14, 7, 2, 0, 0, 0]
Prethodna kombinacija: prosek=1.2188; pogodaka 0..7=[13, 30, 16, 4, 1, 0, 0, 0]
Pozadina bez pobuda: prosek=1.4219; pogodaka 0..7=[10, 27, 19, 6, 2, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 4699: 6,10,12,17,21,27,34
Trosak projekcije=0.9322975687; globalni minimum potvrden.
Jacina dugih pobuda=1.791527; lokalnih pobuda=2.814317
Kanali pobuda su strukturna analogija; trosak rangiranja nije verovatnoca.



Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_2971.csv; redova=2971
Validacija: redovi 2860..2907; alpha_long=100; alpha_local=1; prosek=1.3542
Test: redovi 2908..2971
LANDAU V3 SUPERFLUIDITY HE II: prosek=1.2969; pogodaka 0..7=[14, 24, 19, 7, 0, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[11, 30, 16, 7, 0, 0, 0, 0]
Pozadina bez pobuda: prosek=1.2812; pogodaka 0..7=[15, 23, 19, 7, 0, 0, 0, 0]
Poslednji red: 18,20,21,24,29,34,38
SLEDECI RED 2972: 4,9,11,17,20,28,35
Trosak projekcije=1.3394278419; globalni minimum potvrden.
Jacina dugih pobuda=1.862265; lokalnih pobuda=2.792055
Kanali pobuda su strukturna analogija; trosak rangiranja nije verovatnoca.



Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_plus_1727.csv; redova=1727
Validacija: redovi 1616..1663; alpha_long=1; alpha_local=1; prosek=1.5625
Test: redovi 1664..1727
LANDAU V3 SUPERFLUIDITY HE II: prosek=1.3438; pogodaka 0..7=[12, 24, 23, 4, 1, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[15, 23, 18, 8, 0, 0, 0, 0]
Pozadina bez pobuda: prosek=1.2656; pogodaka 0..7=[12, 27, 21, 4, 0, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 1728: 3,9,14,19,24,30,34
Trosak projekcije=1.3285711428; globalni minimum potvrden.
Jacina dugih pobuda=1.287900; lokalnih pobuda=2.157951
Kanali pobuda su strukturna analogija; trosak rangiranja nije verovatnoca.
"""



"""
Ideja elementarnih pobuđenja iz teorije helijuma II. Promene razmaka opisuje kroz duge talase i lokalne deformacije.

Ideja je razlaganje rasporeda na sporu pozadinu, duge pobude i lokalne pobude, uz učenje njihove sprege — strukturna primena Landauovog pristupa elementarnim pobuđenjima.


Na poslednja 64 kola:

Metod	Prosečan broj pogodaka
V3	    1,3594

Prethodna kombinacija	1,2188
Spora pozadina bez pobuda	1,4219


V3 je na ovom testu nadmašila prethodnu kombinaciju, ali ne i samu pozadinu. Provereni su rekonstrukcija, očuvanje raspona, hronologija, ponovljivost i globalni minimum troška izbora kombinacije.
"""
