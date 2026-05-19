from typing import Any

from app.config import BaseConfig as Settings
from app.utils.common import (
    get_all_check_for_device,
    get_region_override_commands,
    get_region_override_data,
)
from app.utils.hostgroups_override import get_hostgroup


def build_device_context(
    device,
    commands_dict: dict,
    device_brand_checks_list,
    device_dashboard_override: dict,
    device_brand_region: str,
) -> dict[str, Any]:
    """
    Mirror the logic of get_device_host_object() from nornir_inventory_manager.
    Returns a plain dict instead of a Nornir Host object.
    """
    brand_model = device.brand_model
    device_type = device.device_type
    name = device.device_name
    dashboard = device.dashboard

    only_ping_check = True
    brand_category = None
    commands = None
    username = None
    password = None

    if brand_model in commands_dict:
        brand_model_category = commands_dict[brand_model]["brand_category"]

        if device_brand_region != "PARIS":
            username = Settings.DEVICE_CREDENTIALS.get(device_brand_region, {}).get("DEVICE_USERNAME")
            password = Settings.DEVICE_CREDENTIALS.get(device_brand_region, {}).get("DEVICE_PASSWORD")
        else:
            if brand_model_category in Settings.DEVICE_CREDENTIALS_BRAND_BASE:
                username = Settings.DEVICE_CREDENTIALS_BRAND_BASE[brand_model_category].get("DEVICE_USERNAME")
                password = Settings.DEVICE_CREDENTIALS_BRAND_BASE[brand_model_category].get("DEVICE_PASSWORD")
            else:
                username = Settings.DEVICE_CREDENTIALS.get(device_brand_region, {}).get("DEVICE_USERNAME")
                password = Settings.DEVICE_CREDENTIALS.get(device_brand_region, {}).get("DEVICE_PASSWORD")

        override_brand = device_dashboard_override.get(device_type)
        if override_brand:
            commands = override_brand
            brand_category = "override_" + brand_model
        else:
            commands = get_all_check_for_device(
                commands_dict[brand_model]["checks"],
                device_brand_checks_list,
            )
            brand_category = commands_dict[brand_model]["brand_category"]

        override_region = get_region_override_data().get(dashboard)
        if override_region:
            commands, brand_category = get_region_override_commands(
                commands,
                dashboard,
                name,
                brand_model,
                brand_category,
            )

        # Keep original behavior: only ping if commands is None.
        only_ping_check = commands is None

        host_group = get_hostgroup(name, dashboard, brand_category)
        if host_group is not None and commands is not None:
            for chk in host_group.checks_to_exclude:
                try:
                    commands.pop(chk)
                except KeyError:
                    pass
        else:
            host_group = None

    return {
        "device_name": name,
        "device_ip": device.device_ip,
        "port": device.port,
        "dashboard": dashboard,
        "assert_id": device.assert_id,
        "infra_type": device.infra_type,
        "infra_name": device.infra_name,
        "brand_model": brand_model,
        "device_id": device.id,
        "device_type": device_type,
        "os_type": device.os_type,
        "brand_category": brand_category,
        "only_ping_check": only_ping_check,
        "region": device_brand_region,
        "username": username,
        "password": password,
        "host_group": host_group,
        "checks": {"commands": commands},
    }


def detect_platform(ctx: dict[str, Any]) -> str:
    """Mirror fetch_host_platform() logic from device_helper."""
    from app.utils.constants import (
        ARISTA_SERIES,
        CHECK_POINT_SERIES,
        CITRIX_SERIES,
        F5_NETWORK_SERIES,
        SKYHIGH_SERIES,
        FORTINET_SERIES,
    )

    brand_category = ctx.get("brand_category", "")
    os_type = ctx.get("os_type", "")

    if brand_category in FORTINET_SERIES:
        return "fortinet"
    if brand_category in ARISTA_SERIES:
        return "cisco_ios" if os_type == "mos" else "arista_eos"
    if brand_category in CITRIX_SERIES:
        return "netscaler"
    if brand_category in CHECK_POINT_SERIES:
        return "cisco_ios"
    if brand_category in F5_NETWORK_SERIES:
        return "f5_tmsh"
    if brand_category in SKYHIGH_SERIES:
        return "linux"

    if os_type == "nx-os":
        return "cisco_nxos"
    if os_type == "ios-xe":
        return "cisco_xe"
    if os_type in ("ios-xrv", "ios-xr"):
        return "cisco_xr"
    return "cisco_ios"
