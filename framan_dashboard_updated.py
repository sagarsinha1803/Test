import json
import logging

from framan_dashboard_async import run_framan_dashboard, run_framan_dashboard_sync

__all__ = ["run_framan_dashboard", "run_framan_dashboard_sync"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    results = run_framan_dashboard_sync(
        dashboards=["dashboard_pcn"],
        connect_timeout=15,
        read_timeout=60,
        device_concurrency=10,
        bastion_max_connections=10,
        ping_timeout=3,
        ping_before_connect=True,
        # Example: fail a known slow check fast instead of waiting 60s x 3 retries
        # check_timeout_overrides={"Problem Check Name": 10},
        # check_retry_overrides={"Problem Check Name": 1},
        # Example: run only a few devices
        device_names=[],
        # Example: regex for FW devices
        # device_name_regex=r"^PW4PFW",
    )

    print(json.dumps(results, indent=2, default=str))
