import asyncio
import logging
from traceback import format_exc
from typing import Any, Optional

import asyncssh

from app.config import BaseConfig as Settings
from app.dal.nornir_dal import NornirDal
from app.utils.common import (
    get_brand_model_pattern_config,
    get_brand_models_commands,
    get_device_dashboard_config,
    get_device_override_config,
    remove_pinged_check_brand_model,
)

from .device_context import build_device_context
from .device_filter import filter_devices
from .device_worker import device_worker
from .result_builder import build_unreachable_result

logger = logging.getLogger(__name__)


async def run_framan_dashboard(
    dashboards: list[str],
    connect_timeout: int = 15,
    read_timeout: int = 60,
    device_concurrency: int = 10,
    bastion_max_connections: int = 10,
    ping_timeout: int = 5,
    ping_before_connect: bool = True,
    # per-check performance overrides
    check_timeout_overrides: Optional[dict[str, int]] = None,
    check_retry_overrides: Optional[dict[str, int]] = None,
    # device filter options
    device_names: Optional[list[str]] = None,
    device_ips: Optional[list[str]] = None,
    device_name_regex: Optional[str] = None,
    exclude_names: Optional[list[str]] = None,
    exclude_ips: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """
    Async equivalent of NornirHelper.get_health_status_check() for fra/man dashboards.

    Flow:
      1. Load inventory from DB
      2. Ping each device via bastion
      3. SSH tunnel through bastion
      4. Execute show commands
      5. Return list of device payloads
    """

    # Load inventory/config.
    device_dashboard_config = get_device_dashboard_config()
    brand_model_with_checks = remove_pinged_check_brand_model(device_dashboard_config)
    device_brand_config = NornirDal.get_device_barand_model_config()
    device_dashboard_override = get_device_override_config()
    brand_model_config = get_brand_model_pattern_config(device_brand_config)
    commands_dict = get_brand_models_commands(
        brand_model_with_checks,
        brand_model_config,
    )
    list_of_devices = NornirDal.get_all_devices(dashboards)

    # Optional filtering.
    before = len(list_of_devices)
    list_of_devices = filter_devices(
        list_of_devices,
        device_names=device_names,
        device_ips=device_ips,
        device_name_regex=device_name_regex,
        exclude_names=exclude_names,
        exclude_ips=exclude_ips,
    )
    after = len(list_of_devices)

    if device_names or device_ips or device_name_regex or exclude_names or exclude_ips:
        logger.info(
            f"Device filter applied: {before} -> {after} devices "
            f"(names={device_names}, ips={device_ips}, "
            f"regex={device_name_regex}, "
            f"exclude_names={exclude_names}, exclude_ips={exclude_ips})"
        )

    logger.info(
        f"Loaded {len(list_of_devices)} devices for dashboards: {dashboards}"
    )

    # Group devices by region.
    region_device_map: dict[str, list[dict[str, Any]]] = {}

    for device in list_of_devices:
        device_brand_checks_list = (
            device_dashboard_config
            .get(device.dashboard, {})
            .get(device.infra_type, {})
            .get(device.infra_name, {})
            .get(device.brand_model)
        )
        device_brand_region = (
            device_dashboard_config
            .get(device.dashboard, {})
            .get(device.infra_type, {})
            .get(device.infra_name, {})
            .get("region")
        )

        if not device_brand_region:
            logger.warning(
                f"No region found for device {device.device_name}, skipping"
            )
            continue

        ctx = build_device_context(
            device,
            commands_dict,
            device_brand_checks_list,
            device_dashboard_override,
            device_brand_region,
        )
        region_device_map.setdefault(device_brand_region, []).append(ctx)

    all_results: list[dict[str, Any]] = []

    # Process each region (one bastion connection per region).
    for region, device_contexts in region_device_map.items():
        bastion_host = Settings.DEVICE_CREDENTIALS[region]["JUMPHOST_IP"]
        bastion_user = Settings.DEVICE_CREDENTIALS[region]["JUMPHOST_USER"]
        bastion_port = int(Settings.DEVICE_CREDENTIALS[region].get("JUMPHOST_PORT", 22))
        bastion_key = (
            Settings.DEVICE_CREDENTIALS[region].get("KEY_PATH")
            or Settings.DEVICE_RSA_FILE
        )

        logger.info(
            f"Connecting to bastion [{region}] {bastion_host}:{bastion_port} "
            f"for {len(device_contexts)} devices"
        )

        try:
            async with asyncssh.connect(
                host=bastion_host,
                port=bastion_port,
                username=bastion_user,
                client_keys=[bastion_key],
                known_hosts=None,
                connect_timeout=connect_timeout,
            ) as bastion_conn:

                effective_device_concurrency = max(
                    1,
                    min(device_concurrency, bastion_max_connections),
                )
                if effective_device_concurrency != device_concurrency:
                    logger.info(
                        f"Capping device concurrency from {device_concurrency} "
                        f"to bastion limit {effective_device_concurrency}"
                    )

                semaphore = asyncio.Semaphore(effective_device_concurrency)

                tasks = [
                    device_worker(
                        ctx,
                        bastion_conn,
                        semaphore,
                        connect_timeout,
                        read_timeout,
                        ping_timeout=ping_timeout,
                        ping_before_connect=ping_before_connect,
                        check_timeout_overrides=check_timeout_overrides,
                        check_retry_overrides=check_retry_overrides,
                    )
                    for ctx in device_contexts
                ]

                region_results = await asyncio.gather(
                    *tasks,
                    return_exceptions=False,
                )
                all_results.extend(region_results)

        except Exception as ex:
            logger.error(
                f"Failed to connect to bastion for region {region}: "
                f"{ex}\n{format_exc()}"
            )
            for ctx in device_contexts:
                all_results.append(
                    build_unreachable_result(
                        ctx,
                        f"Jumphost not reachable: {ex}",
                    )
                )

    logger.info(
        f"Fra/Man dashboard complete. "
        f"Total devices processed: {len(all_results)}"
    )
    return all_results


def run_framan_dashboard_sync(
    dashboards: list[str],
    connect_timeout: int = 15,
    read_timeout: int = 60,
    device_concurrency: int = 10,
    bastion_max_connections: int = 10,
    ping_timeout: int = 5,
    ping_before_connect: bool = True,
    # per-check performance overrides
    check_timeout_overrides: Optional[dict[str, int]] = None,
    check_retry_overrides: Optional[dict[str, int]] = None,
    # device filter options
    device_names: Optional[list[str]] = None,
    device_ips: Optional[list[str]] = None,
    device_name_regex: Optional[str] = None,
    exclude_names: Optional[list[str]] = None,
    exclude_ips: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    return asyncio.run(
        run_framan_dashboard(
            dashboards=dashboards,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            device_concurrency=device_concurrency,
            bastion_max_connections=bastion_max_connections,
            ping_timeout=ping_timeout,
            ping_before_connect=ping_before_connect,
            check_timeout_overrides=check_timeout_overrides,
            check_retry_overrides=check_retry_overrides,
            device_names=device_names,
            device_ips=device_ips,
            device_name_regex=device_name_regex,
            exclude_names=exclude_names,
            exclude_ips=exclude_ips,
        )
    )
