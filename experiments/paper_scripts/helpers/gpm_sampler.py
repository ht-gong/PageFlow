"""One GPU's utilization and PCIe traffic from NVML's GPM metrics, as CSV on
stdout until SIGTERM or SIGINT.

    python gpm_sampler.py <gpu uuid> <interval s>

Each row covers the interval that ends at its ``time`` (epoch seconds):
``sm_util`` (% of SMs busy), ``sm_occupancy`` (% of warp slots resident),
``gr_util`` (% of time the graphics/compute engine was busy), ``dram_bw_util``
(% of peak), ``pcie_rx_mibs`` / ``pcie_tx_mibs`` (MiB/s into / out of the GPU:
host->device and device->host), with ``mem_used_mib`` and ``procs`` (compute
processes on the GPU) read at its end.  A metric NVML does not return is empty.

GPM is NVML's own sampling, so unlike Nsight's GPU metrics or ncu it does not
need the performance-counter permission (RmProfilingAdminOnly) and runs in an
unprivileged container.  It needs ``pynvml`` (``nvidia-ml-py``, in
requirements-dev.txt, so in the engine image and the native environment) and
a Hopper or later GPU.  ``e4_ablation_profile.py`` runs it beside each
profiled process, with the engine's python (its ``--python``).
"""
import math
import signal
import sys
import time

import pynvml as N

METRICS = [("sm_util", "NVML_GPM_METRIC_SM_UTIL"),
           ("sm_occupancy", "NVML_GPM_METRIC_SM_OCCUPANCY"),
           ("gr_util", "NVML_GPM_METRIC_GRAPHICS_UTIL"),
           ("dram_bw_util", "NVML_GPM_METRIC_DRAM_BW_UTIL"),
           ("pcie_rx_mibs", "NVML_GPM_METRIC_PCIE_RX_PER_SEC"),
           ("pcie_tx_mibs", "NVML_GPM_METRIC_PCIE_TX_PER_SEC")]


def main():
    uuid, interval = sys.argv[1], float(sys.argv[2])
    stop = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.append(1))
    N.nvmlInit()
    h = N.nvmlDeviceGetHandleByUUID(uuid)
    if not N.nvmlGpmQueryDeviceSupport(h).isSupportedDevice:
        sys.exit(f"gpm_sampler: {uuid} does not support GPM metrics")
    m = N.c_nvmlGpmMetricsGet_t()
    m.version = N.NVML_GPM_METRICS_GET_VERSION
    m.numMetrics = len(METRICS)
    for i, (_, name) in enumerate(METRICS):
        m.metrics[i].metricId = getattr(N, name)
    prev, cur = N.nvmlGpmSampleAlloc(), N.nvmlGpmSampleAlloc()
    N.nvmlGpmSampleGet(h, prev)
    print("time,mem_used_mib,procs," + ",".join(c for c, _ in METRICS), flush=True)
    t_next = time.time()
    while not stop:
        t_next += interval
        time.sleep(max(0.0, t_next - time.time()))
        N.nvmlGpmSampleGet(h, cur)
        m.sample1, m.sample2 = prev, cur
        N.nvmlGpmMetricsGet(m)
        # A metric can come back NaN with NVML_SUCCESS (DRAM bandwidth does).
        vals = [f"{m.metrics[i].value:.1f}" if m.metrics[i].nvmlReturn == 0
                and math.isfinite(m.metrics[i].value) else ""
                for i in range(len(METRICS))]
        print(f"{time.time():.3f},{N.nvmlDeviceGetMemoryInfo(h).used >> 20},"
              f"{len(N.nvmlDeviceGetComputeRunningProcesses(h))},"
              + ",".join(vals), flush=True)
        prev, cur = cur, prev


if __name__ == "__main__":
    main()
