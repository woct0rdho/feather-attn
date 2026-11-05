"""Benchmark protocol for the kernels in this package.

One provider measurement is: warmup, then `rounds` rounds of `iters` back-to-back launches inside one
event pair, reported as the median over rounds of the per-launch milliseconds plus the p20/p80 spread.
Providers in a case are interleaved and their order alternates between rounds, so the sustained-clock
drift of this machine hits every provider alike and the ratio against the control is a paired median
of per-round ratios instead of a ratio of medians measured minutes apart.

Why not `triton.testing.do_bench`:
- it flushes L2 between iterations, so every launch is timed cold. That is right for a one-shot
  streaming kernel and wrong for a model that calls the same weight repeatedly: the same tile measures
  22.9 TFLOPS cold and 32.9 warm on `4096 x 2048 x 2048`. `cold=True` keeps that regime for the cases
  where it is the real one, instead of making it the default.
- it sizes its warmup by wall time, so a 100 ms kernel gets one warmup launch at best, and it sizes its
  repetition the same way, so iteration counts swing by orders of magnitude across shapes.
- it times one provider at a time, so a clock change between providers looks like a difference between
  them.
- it reports a single number, so a 1% change cannot be separated from noise.
"""

import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Measurement:
    """One provider on one case."""

    median_ms: float
    p20_ms: float
    p80_ms: float
    samples_ms: tuple[float, ...]

    @property
    def spread(self) -> float:
        """Relative p20-p80 width, i.e. how much of a difference this measurement can resolve."""
        return (self.p80_ms - self.p20_ms) / self.median_ms if self.median_ms > 0 else 0.0


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] * (1.0 - (position - lower)) + ordered[upper] * (position - lower)


def _timed_batch(fn: Callable[[], object], iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def _timed_one(fn: Callable[[], object]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


_FLUSH = None


def _flush_l2() -> None:
    global _FLUSH
    if _FLUSH is None:
        # 64 MB is 32x the 2 MB L2, so one pass evicts everything that was resident.
        _FLUSH = torch.empty(64 << 20, dtype=torch.int32, device="cuda")
    _FLUSH.zero_()


def time_fn(
    fn: Callable[[], object],
    *,
    warmup: int = 5,
    rounds: int = 5,
    target_ms: float = 50.0,
    max_iters: int = 32,
    cold: bool = False,
) -> Measurement:
    """Time one provider. `target_ms` sets how long one round should take, so the iteration count
    adapts from a 20 us kernel to a 200 ms one."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    estimate_ms = max(_timed_batch(fn, 1), 1e-6)
    iters = min(max(round(target_ms / estimate_ms), 1), max_iters)

    samples = []
    if cold:
        for _ in range(rounds * iters):
            _flush_l2()
            samples.append(_timed_one(fn))
    else:
        for _ in range(rounds):
            samples.append(_timed_batch(fn, iters))

    median_ms = statistics.median(samples)
    return Measurement(
        median_ms=median_ms,
        p20_ms=percentile(samples, 0.2),
        p80_ms=percentile(samples, 0.8),
        samples_ms=tuple(samples),
    )


def measure(
    providers: dict[str, Callable[[], object]],
    *,
    warmup: int = 5,
    rounds: int = 5,
    target_ms: float = 50.0,
    max_iters: int = 32,
    cold: bool = False,
) -> dict[str, Measurement]:
    """Time several providers on one case, interleaved, with the order rotated per round.

    Rotating rather than reversing is what makes the pairing honest: with `n` providers and `r` rounds, a
    provider sits at each position `floor(r / n)` times (more, up to one extra, when `r % n`), whereas a
    reversal leaves the middle position owned by the same provider in every round. That bias is not
    hypothetical - two providers that launch the same kernel measured 1.6% apart when one of them always
    sat in the middle of a three-provider run.
    """
    names = list(providers)
    per_round: dict[str, list[float]] = {name: [] for name in names}

    if cold:
        for name in names:
            for _ in range(warmup):
                providers[name]()
        torch.cuda.synchronize()
        iters = 1
        for _ in range(rounds * max_iters):
            for name in names:
                _flush_l2()
                per_round[name].append(_timed_one(providers[name]))
    else:
        estimates = {}
        for name in names:
            for _ in range(warmup):
                providers[name]()
        torch.cuda.synchronize()
        for name in names:
            estimates[name] = max(_timed_batch(providers[name], 1), 1e-6)
        iters = {name: min(max(round(target_ms / ms), 1), max_iters) for name, ms in estimates.items()}
        for index in range(rounds):
            shift = index % len(names)
            order = names[shift:] + names[:shift]
            for name in order:
                per_round[name].append(_timed_batch(providers[name], iters[name]))

    return {
        name: Measurement(
            median_ms=statistics.median(samples),
            p20_ms=percentile(samples, 0.2),
            p80_ms=percentile(samples, 0.8),
            samples_ms=tuple(samples),
        )
        for name, samples in per_round.items()
    }


def speedup(candidate: Measurement, control: Measurement) -> float:
    """How many times faster `candidate` is than `control`, from the per-round ratios."""
    if len(candidate.samples_ms) != len(control.samples_ms):
        raise ValueError("paired ratios need the same number of samples")
    return statistics.median(w / c for w, c in zip(control.samples_ms, candidate.samples_ms))


@dataclass
class Case:
    """One shape (or one operator state) with the providers to compare on it."""

    label: str
    providers: dict[str, Callable[[], object]]
    scale: float = 1.0
    control: str | None = None


def run_cases(
    cases: list[Case],
    *,
    unit: str = "TFLOPS",
    control: str | None = None,
    warmup: int = 5,
    rounds: int = 5,
    target_ms: float = 50.0,
    max_iters: int = 32,
    cold: bool = False,
    json_path: str | None = None,
) -> list[dict]:
    """Time every case, print one row per provider, optionally write the raw samples as JSON.

    `scale` converts milliseconds to the reported unit: GFLOP for TFLOPS, bytes / 1e6 for GB/s.
    `control` names the provider the ratios are taken against. It defaults to the case's own override
    and then to the first provider.
    """
    print(f"{'case':34s} {'provider':22s} {'ms':>9s} {'p20':>9s} {'p80':>9s} {'spread':>7s} {'vs ctl':>7s} {unit:>8s}")
    records = []
    for case in cases:
        measurements = measure(case.providers, warmup=warmup, rounds=rounds, target_ms=target_ms, max_iters=max_iters, cold=cold)
        control_name = case.control or control or next(iter(case.providers))
        case_record = {"case": case.label, "providers": {}}
        for name, measurement in measurements.items():
            value = case.scale / measurement.median_ms if measurement.median_ms > 0 else float("nan")
            ratio = speedup(measurement, measurements[control_name]) if name != control_name else 1.0
            print(
                f"{case.label:34s} {name:22s} {measurement.median_ms:.3f} {measurement.p20_ms:.3f} "
                f"{measurement.p80_ms:.3f} {100.0 * measurement.spread:.3f}% {ratio:.3f} {value:.3f}"
            )
            case_record["providers"][name] = {
                "median_ms": measurement.median_ms,
                "p20_ms": measurement.p20_ms,
                "p80_ms": measurement.p80_ms,
                "samples_ms": list(measurement.samples_ms),
                "value": value,
                "speedup_vs_control": ratio,
            }
        records.append(case_record)

    if json_path:
        payload = {
            "protocol": {
                "warmup": warmup,
                "rounds": rounds,
                "target_ms": target_ms,
                "max_iters": max_iters,
                "cold": cold,
                "control": control,
                "unit": unit,
            },
            "cases": records,
        }
        with open(json_path, "w") as handle:
            json.dump(payload, handle, indent=1)
        print(f"wrote {json_path}")
    return records


def add_protocol_args(parser) -> None:
    """The flags every benchmark in bench/ shares."""
    parser.add_argument("--warmup", type=int, default=5, help="launches before timing starts")
    parser.add_argument("--rounds", type=int, default=5, help="timing rounds. The reported number is their median")
    parser.add_argument("--target-ms", type=float, default=50.0, help="how long one round should take")
    parser.add_argument("--json", dest="json_path", default=None, help="write the per-round samples here")
    parser.add_argument("--cold", action="store_true", help="flush L2 between launches (one-shot streaming regime)")
