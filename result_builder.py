from typing import Any

from app.utils.constants import HealthStatus


def build_device_result(
    ctx: dict[str, Any],
    result_dict: dict[str, Any],
) -> dict[str, Any]:
    return {
        "device_name": ctx["device_name"],
        "device_ip": ctx["device_ip"],
        "assert_id": ctx.get("assert_id"),
        "dashboard": ctx.get("dashboard"),
        "port": ctx["port"],
        "brand_model": ctx.get("brand_model"),
        "infra_name": ctx.get("infra_name"),
        "infra_type": ctx.get("infra_type"),
        "device_json_data": {ctx["device_name"]: result_dict},
    }


def build_unreachable_result(
    ctx: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    checks = ctx["checks"]["commands"] or {}
    result_dict: dict[str, Any] = {}

    for each_check in checks:
        result_dict[each_check] = {
            "status": HealthStatus.NOTCONNECTED.value,
            "output": None,
            "err": reason,
        }

    result_dict["Ping Status"] = {
        "status": "NOTREACHABLE",
        "output": None,
        "err": reason,
    }
    return build_device_result(ctx, result_dict)


def build_only_ping_result(ctx: dict[str, Any]) -> dict[str, Any]:
    return build_device_result(
        ctx,
        {"Ping Status": {"status": "REACHABLE", "output": None, "err": None}},
    )
