"""Observe AIPerf phases and label the scope of optional backend scrapes."""

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from threading import Event, Thread
import time

from server_metrics import capture, compact_sample, delta, request_count, window_metrics


_POLL_SECONDS = 0.15
_HEARTBEAT_SECONDS = 30
_WINDOW_TOLERANCE_NS = 5_000_000_000
_PHASE = re.compile(
    r"Phase [\w. -]+ \((warmup|profiling)\) (started|sending complete|complete) \| ([^\n]+)"
)
_DATASET = re.compile(r"Dataset configured in ([\d.]+) seconds")
_PROGRESS = re.compile(r"Phase (warmup|profiling) progress \| ([^\n]+)")
_REALTIME = re.compile(r"\b(rps=\S+[^\n]*?\bdone=[\d,]+\s+ok=[\d,]+\s+err=[\d,]+)\b")
_NATIVE_FIELDS = {
    "start_ns": "profile_start_ns",
    "sent_end_ns": "send_end_ns",
    "requests_end_ns": "requests_end_ns",
    "final_requests_completed": "final_requests_completed",
    "final_requests_cancelled": "final_requests_cancelled",
    "final_request_errors": "final_request_errors",
    "grace_period_timeout_triggered": "grace_period_timeout_triggered",
    "timeout_triggered": "send_stop_timeout_triggered",
    "was_cancelled": "native_cancelled",
}


def write_json_atomic(path, value):
    part = path.with_suffix(path.suffix + ".tmp")
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    part.replace(path)


def _fields(text):
    values = {}
    for name, raw in re.findall(r"\b([a-z_]+)=(-?\d+|True|False|None)\b", text):
        values[name] = {"True": True, "False": False, "None": None}.get(
            raw, int(raw) if raw.lstrip("-").isdigit() else None,
        )
    return values


def _error_category(record, metadata):
    if metadata.get("context_overflow_skip"):
        return "context_overflow"
    if metadata.get("was_cancelled"):
        return "cancelled"
    error = record.get("error")
    if not error:
        return None
    if not isinstance(error, dict):
        return "other"
    kind = str(error.get("type") or "").casefold()
    if "cancelled" in kind or "canceled" in kind:
        return "cancelled"
    if "timeout" in kind or "timedout" in kind:
        return "timeout"
    if "contextoverflow" in kind.replace("_", ""):
        return "context_overflow"
    code = error.get("code")
    if isinstance(code, int) and 400 <= code <= 599:
        return f"http_{code}"
    return "other"


@dataclass
class _RequestEvidence:
    count: int = 0
    bounded_count: int = 0
    first_start_ns: int | None = None
    last_end_ns: int | None = None
    last_success_ns: int | None = None
    errors: Counter = field(default_factory=Counter)

    def add(self, record, metadata):
        self.count += 1
        if category := _error_category(record, metadata):
            self.errors[category] += 1
        start = metadata.get("request_start_ns")
        if type(start) is int and start > 0:
            self.first_start_ns = min(self.first_start_ns or start, start)
        end = metadata.get("request_end_ns")
        if type(start) is int and type(end) is int and 0 < start <= end:
            self.bounded_count += 1
        if type(end) is int and end > 0:
            self.last_end_ns = max(self.last_end_ns or end, end)
            if not category:
                self.last_success_ns = max(self.last_success_ns or end, end)


class PhaseObserver:
    def __init__(self, artifact_dir: Path, point_log: Path, *,
                 logical_phase: str, experiment: str, metrics_url: str | None):
        self.artifact_dir = Path(artifact_dir)
        self.point_log = Path(point_log)
        self.logical_phase = logical_phase
        self.experiment = experiment
        self.metrics_url = metrics_url
        self.returncode = None
        self.summary = {
            **dict.fromkeys(_NATIVE_FIELDS.values()),
            "experiment": experiment,
            "logical_phase": logical_phase,
            "state": "starting",
            "in_flight_at_cutoff": None,
            "dataset_seconds": None,
            "warmup_seconds": None,
            "process_start_ns": None,
            "process_end_ns": None,
            "native_events": {},
            "sample_times": {},
            "observation_errors": [],
            "artifacts": str(self.artifact_dir),
        }
        self._snapshots = {}
        self._records = {}
        self._records_valid = False
        self._stop = Event()
        self._offset = 0
        self._next_heartbeat = 0
        self._thread = Thread(target=self._monitor, name="aiperf-observer", daemon=True)

    def __enter__(self):
        self._sample("before")
        self.summary["process_start_ns"] = time.time_ns()
        self.summary["state"] = "configuring"
        self._status()
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.summary["process_end_ns"] = time.time_ns()
        self._stop.set()
        self._thread.join()
        self._sample("after")
        self._read_request_evidence()
        self.summary["returncode"] = self.returncode
        self.summary["state"] = (
            "interrupted" if isinstance(exc, KeyboardInterrupt) or self.returncode in (130, -2) else
            "error" if exc_type or self.returncode not in (None, 0) else "complete"
        )
        if exc_type:
            self.summary["process_error_type"] = exc_type.__name__
        start, send, end = (self.summary[key] for key in
                            ("profile_start_ns", "send_end_ns", "requests_end_ns"))
        for key, first, last in (("profile_send_seconds", start, send), ("drain_seconds", send, end)):
            if isinstance(first, int) and isinstance(last, int) and last >= first:
                self.summary[key] = (last - first) / 1e9
        self._publish()

    def _read_request_evidence(self):
        path = self.artifact_dir / "profile_export.jsonl"
        invalid = 0
        try:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        metadata = record["metadata"]
                        phase = metadata.get("phase_kind") or metadata.get("benchmark_phase")
                    except (ValueError, TypeError, KeyError, AttributeError):
                        invalid += 1
                        continue
                    if phase in ("warmup", "profiling"):
                        self._records.setdefault(phase, _RequestEvidence()).add(record, metadata)
        except FileNotFoundError:
            return
        except (OSError, UnicodeError) as exc:
            self.summary["observation_errors"].append(f"request evidence: {exc}")
            return
        self._records_valid = invalid == 0
        if invalid:
            self.summary["observation_errors"].append(f"request evidence: {invalid} invalid records")
        warmup = self._records.get("warmup")
        if warmup is not None:
            self.summary["warmup_request_error_types"] = dict(warmup.errors)
            self.summary["warmup_last_response_ns"] = warmup.last_end_ns
        formal = self._records.get("profiling")
        if formal is not None:
            self.summary["request_error_types"] = dict(formal.errors)
            self.summary["first_request_start_ns"] = formal.first_start_ns
            self.summary["last_response_ns"] = formal.last_success_ns
            send_end = self.summary["send_end_ns"]
            if formal.last_success_ns is not None and send_end is not None:
                self.summary["drain_last_response_seconds"] = max(0, formal.last_success_ns - send_end) / 1e9

    def _write(self, path, content):
        try:
            write_json_atomic(path, content)
        except OSError as exc:
            error = f"{path.name}: {exc}"
            errors = self.summary["observation_errors"]
            if error not in errors:
                errors.append(error)

    def _status(self):
        self._write(self.point_log / "status.json", {
            **self.summary, "observed_ns": time.time_ns(),
        })

    def _publish(self):
        self._write(self.point_log / f"{self.logical_phase}.phase.json", self.summary)
        self._status()

    def _sample(self, label, *, periodic=False):
        if not self.metrics_url:
            return
        stem = "metrics" if self.logical_phase == "formal" else f"{self.logical_phase}.metrics"
        path = None if periodic else self.point_log / f"{stem}-{label}.prom"
        timing = {"sample": label, "start_ns": time.time_ns()}
        try:
            snapshot = capture(self.metrics_url, path)
        except OSError as exc:
            self.summary["observation_errors"].append(f"{label}: {exc}")
            snapshot = None
        timing.update(end_ns=time.time_ns(), ok=snapshot is not None)
        if not periodic:
            self.summary["sample_times"][label] = timing
            if snapshot is not None:
                self._snapshots[label] = snapshot
        if snapshot is None:
            self.summary["observation_errors"].append(f"Prometheus scrape failed: {label}")
        try:
            with (self.point_log / f"{self.logical_phase}.metrics.jsonl").open("a", encoding="utf-8") as stream:
                compact = compact_sample(snapshot) if snapshot is not None else []
                stream.write(json.dumps({**timing, "metrics": compact}, ensure_ascii=False) + "\n")
        except OSError as exc:
            self.summary["observation_errors"].append(f"metrics history: {exc}")

    def _observe_line(self, line):
        if dataset := _DATASET.search(line):
            self.summary["dataset_seconds"] = float(dataset.group(1))
        if progress := _PROGRESS.search(line):
            self.summary["latest_progress"] = progress.group(2)
        elif progress := _REALTIME.search(line):
            self.summary["latest_progress"] = progress.group(1)
        if "PhaseRecordsStats(" in line and "phase_kind='profiling'" in line:
            values = _fields(line)
            for native, name in _NATIVE_FIELDS.items():
                if values.get(native) is not None:
                    self.summary[name] = values[native]
        matched = _PHASE.search(line)
        if not matched:
            return None
        kind, event, text = matched.groups()
        if event == "started":
            self.summary.pop("latest_progress", None)
        values = _fields(text)
        self.summary["native_events"][f"{kind}_{event.replace(' ', '_')}"] = {
            "observed_ns": time.time_ns(), **values,
        }
        if kind == "warmup":
            self.summary["state"] = "warmup"
            if event == "complete":
                elapsed = re.search(r"\belapsed=([\d.]+)s", text)
                if elapsed:
                    self.summary["warmup_seconds"] = float(elapsed.group(1))
                return "window-start" if self.experiment == "P03" else None
            return None
        if event == "started":
            self.summary["state"] = "profiling"
        elif event == "sending complete":
            self.summary["state"] = "draining"
            self.summary["in_flight_at_cutoff"] = values.get("in_flight")
            return "send-end"
        else:
            self.summary["state"] = "exporting"
            for field, output in (("completed", "final_requests_completed"),
                                  ("cancelled", "final_requests_cancelled"),
                                  ("errors", "final_request_errors")):
                if values.get(field) is not None and self.summary[output] is None:
                    self.summary[output] = values[field]
            return "drain-end"
        return None

    def _read(self, *, final=False):
        path = self.point_log / f"{self.logical_phase}.log"
        labels = []
        try:
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                stream.seek(self._offset)
                while line := stream.readline():
                    if not final and not line.endswith("\n"):
                        break
                    self._offset = stream.tell()
                    if label := self._observe_line(line):
                        labels.append(label)
        except FileNotFoundError:
            return
        except OSError as exc:
            self.summary["observation_errors"].append(f"native log: {exc}")
            return
        for label in labels:
            self._status()
            if self.experiment != "P03":
                continue
            if label == "send-end" and "drain-end" in labels:
                continue
            self._sample(label)
            if label == "drain-end" and "send-end" in labels:
                sample = self.summary["sample_times"].get("drain-end")
                if sample is not None:
                    self.summary["sample_times"]["send-end"] = {
                        **sample, "sample": "send-end", "shared_with": "drain-end",
                    }
                if "drain-end" in self._snapshots:
                    self._snapshots["send-end"] = self._snapshots["drain-end"]

    def _monitor(self):
        self._next_heartbeat = time.monotonic() + _HEARTBEAT_SECONDS
        while not self._stop.is_set():
            state = self.summary["state"], self.summary.get("latest_progress")
            self._read()
            if (self.summary["state"], self.summary.get("latest_progress")) != state:
                self._status()
            if time.monotonic() >= self._next_heartbeat:
                self._status()
                if self.summary["state"] in {"warmup", "profiling", "draining"}:
                    self._sample("periodic", periodic=True)
                self._next_heartbeat = time.monotonic() + _HEARTBEAT_SECONDS
            self._stop.wait(_POLL_SECONDS)
        self._read(final=True)

    def _window(self, start_label, end_label, model):
        initial, final = (self._snapshots.get(label) for label in (start_label, end_label))
        if initial is None or final is None:
            return {}, None, None
        return (
            window_metrics(initial, final, model),
            request_count(initial, final, model),
            delta(initial, final, "vllm:prompt_tokens", model),
        )

    def _formal_window_verified(self, expected, count, boundaries, row):
        formal = self._records.get("profiling")
        warmup = self._records.get("warmup")
        if not (self._records_valid and formal and warmup and expected == count == formal.count):
            return False
        if formal.bounded_count != formal.count or warmup.bounded_count != warmup.count:
            return False
        if formal.errors or warmup.errors or row.get("failed_requests") or row.get("cancelled_requests"):
            return False
        start = boundaries.get("window-start")
        end = boundaries.get("drain-end")
        if not (start and end and warmup.last_end_ns and formal.first_start_ns and formal.last_end_ns):
            return False
        return (
            warmup.last_end_ns <= start["start_ns"] <= start["end_ns"] <= formal.first_start_ns
            and formal.last_end_ns <= end["start_ns"]
        )

    def enrich(self, row, model):
        values, count, prompt_tokens = self._window("before", "after", model)
        expected = row.get("completed_requests")
        duration = row.get("duration_seconds")
        tpm = row.get("input_tpm")
        expected_tokens = round(tpm * duration / 60) if tpm is not None and duration else None
        clean = (
            expected and count == expected and
            prompt_tokens is not None and prompt_tokens == expected_tokens and
            not row.get("failed_requests") and not row.get("cancelled_requests")
        )
        self.summary["whole_run_requests_seen"] = count
        for key, value in values.items():
            if self.experiment != "P03" and clean:
                if row.get(key) is None:
                    row[key] = value
                    source = "kv_source" if key.startswith("kv_") else "spec_source"
                    sources = (row.get(source) or "").split(" + ")
                    if "prometheus_formal" not in sources:
                        row[source] = " + ".join(filter(None, [*sources, "prometheus_formal"]))
            else:
                row[f"whole_run_{key}"] = value
        if values and (self.experiment == "P03" or not clean):
            source = "prometheus_process_including_warmup" if self.experiment == "P03" else "prometheus_process"
            row["whole_run_scope"] = source
            row["whole_run_source"] = source

        if self.experiment == "P03":
            window, window_count, _ = self._window("window-start", "drain-end", model)
            boundaries = self.summary["sample_times"]
            offsets = {}
            for label, field in (("window-start", "profile_start_ns"), ("drain-end", "requests_end_ns")):
                if label in boundaries and self.summary[field] is not None:
                    endpoint = self.summary[field]
                    times = boundaries[label]
                    offsets[label] = (
                        times["start_ns"] - endpoint, times["end_ns"] - endpoint,
                    )
            self.summary["formal_window_offsets_ns"] = offsets
            self.summary["formal_window_requests_seen"] = window_count
            nearby = len(offsets) == 2 and all(
                abs(offset) <= _WINDOW_TOLERANCE_NS for pair in offsets.values() for offset in pair
            )
            if window and nearby and expected and window_count is not None:
                if abs(window_count - expected) <= max(1, round(expected * 0.05)):
                    for key, value in window.items():
                        row[f"window_{key}"] = value
                    source = (
                        "prometheus_formal_count_verified"
                        if self._formal_window_verified(expected, window_count, boundaries, row)
                        else "prometheus_approximate_formal_and_drain"
                    )
                    row["window_scope"] = source
                    row["window_source"] = source
        self._publish()
