"""Prometheus metrics exported by the FastAPI process."""

import os
import time

from prometheus_client import Counter, Gauge, Histogram
from prometheus_client.metrics_core import CounterMetricFamily, GaugeMetricFamily
from prometheus_client.registry import Collector, REGISTRY

# On Linux, prometheus_client reads /proc and exports process_cpu_seconds_total
# and process_resident_memory_bytes. Windows has no /proc, so this collector
# fills those same metric names and the Grafana queries stay unchanged.

http_requests_total = Counter(
    "http_requests_total",
    "Total API requests.",
    ["method", "endpoint", "status"],
)
prediction_requests_total = Counter(
    "prediction_requests_total",
    "Total employee records submitted for prediction.",
)
attrition_predictions_total = Counter(
    "attrition_predictions_total",
    "Predictions that the employee is likely to leave (Attrition = Yes).",
)
retention_predictions_total = Counter(
    "retention_predictions_total",
    "Predictions that the employee is likely to stay (Attrition = No).",
)
failed_requests_total = Counter(
    "failed_requests_total",
    "API requests that returned an error status.",
    ["endpoint"],
)
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "API request latency in seconds.",
    ["endpoint"],
)
model_available = Gauge(
    "model_available",
    "1 when the trained model pipeline is loaded, otherwise 0.",
)

model_available.set(0)
prediction_requests_total.inc(0)
attrition_predictions_total.inc(0)
retention_predictions_total.inc(0)


def _resident_memory_bytes() -> float:
    if os.name == "nt":
        import ctypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return float(counters.WorkingSetSize)
    return 0.0


class PortableProcessCollector(Collector):
    """Export process CPU and memory when /proc is not available."""

    def collect(self):
        if os.path.isdir("/proc/self"):
            return []
        cpu = CounterMetricFamily(
            "process_cpu_seconds_total",
            "Total user and system CPU time spent in seconds.",
            value=time.process_time(),
        )
        memory = GaugeMetricFamily(
            "process_resident_memory_bytes",
            "Resident memory size in bytes.",
            value=_resident_memory_bytes(),
        )
        return [cpu, memory]


REGISTRY.register(PortableProcessCollector())
