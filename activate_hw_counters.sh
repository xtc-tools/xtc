sudo sysctl kernel.perf_event_paranoid=-1
echo 0 | sudo tee /proc/sys/kernel/nmi_watchdog
