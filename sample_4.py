import asyncio
import logging
import re
from typing import Any, Optional, Iterable
from traceback import format_exc

import asyncssh

from app.config import BaseConfig as Settings
from app.dal.nornir_dal import NornirDal
from app.utils.constants import HealthStatus
from app.utils.common import (
    get_device_dashboard_config,
    get_brand_model_pattern_config,
    get_brand_models_commands,
    get_all_check_for_device,
    get_device_override_config,
    remove_pinged_check_brand_model,
    get_region_override_commands,
    get_region_override_data,
)
from app.utils.network_commands_factory.network_factory import NetworkFactory
from app.utils.hostgroups_override import get_hostgroup

logger = logging.getLogger(__name__)


def _filter_devices(
    devices: list[Any],
    device_names: Optional[Iterable[str]] = None,
    device_ips: Optional[Iterable[str]] = None,
    device_name_regex: Optional[str] = None,
    exclude_names: Optional[Iterable[str]] = None,
    exclude_ips: Optional[Iterable[str]] = None,
) -> list[Any]:
    name_set = {str(x).strip() for x in (device_names or []) if x and str(x).strip()}
    ip_set = {str(x).strip() for x in (device_ips or []) if x and str(x).strip()}
    ex_name_set = {str(x).strip() for x in (exclude_names or []) if x and str(x).strip()}
    ex_ip_set = {str(x).strip() for x in (exclude_ips or []) if x and str(x).strip()}

    name_re = re.compile(device_name_regex) if device_name_regex else None

    def matches_include(d) -> bool:
        if not name_set and not ip_set and not name_re:
            return True

        dn = getattr(d, "device_name", "") or ""
        dip = getattr(d, "device_ip", "") or ""

        return (
            dn in name_set
            or dip in ip_set
            or bool(name_re and name_re.search(dn))
        )

    def matches_exclude(d) -> bool:
        dn = getattr(d, "device_name", "") or ""
        dip = getattr(d, "device_ip", "") or ""
        return dn in ex_name_set or dip in ex_ip_set

    return [d for d in devices if matches_include(d) and not matches_exclude(d)]


def _build_device_context(
    device,
    commands_dict: dict,
    device_brand_checks_list,
    device_dashboard_ovveride: dict,
    device_brand_region: str,
) -> dict[str, Any]:
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

        override_brand = device_dashboard_ovveride.get(device_type)
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

        only_ping_check = commands is None

        host_group = get_hostgroup(name, dashboard, brand_category)
        if host_group is not None and commands is not None:
            commands = dict(commands)
            for chk in host_group.checks_to_exclude:
                commands.pop(chk, None)
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


async def _ping_via_bastion(
    bastion_conn: asyncssh.SSHClientConnection,
    ip: str,
    timeout: int = 5,
) -> bool:
    try:
        r = await asyncio.wait_for(
            bastion_conn.run(f"ping -c 1 -W 1 {ip}", check=False),
            timeout=timeout,
        )
        return r.exit_status == 0
    except Exception:
        return False


_PROMPT_RE = re.compile(r"(?m)(?:^|\n).{0,120}(\s+\(.*\))?\s*[#>$]\s*$")

_ACCEPT_A_PATTERNS = [
    "press 'a' to accept",
    "press a to accept",
    "(press 'a' to accept)",
    "(press a to accept)",
]

_PRESS_ANY_KEY_PATTERNS = [
    "press any key",
    "press enter",
    "hit any key",
    "press return",
    "--more--",
    "-- more --",
    "space for more",
]


async def _handle_interactive_login_banner(
    conn: asyncssh.SSHClientConnection,
    ctx: dict[str, Any],
    timeout: int = 12,
) -> None:
    device_ip = ctx.get("device_ip", "unknown")

    proc = None
    buf = ""
    start = asyncio.get_event_loop().time()

    try:
        proc = await conn.create_process(term_type="vt100")
        await asyncio.sleep(0.5)

        consecutive_prompt_hits = 0

        while True:
            elapsed = asyncio.get_event_loop().time() - start
            if elapsed > timeout:
                logger.warning(f"[{device_ip}] Banner handling timeout after {timeout}s")
                break

            try:
                chunk = await asyncio.wait_for(proc.stdout.read(1024), timeout=1.0)
            except asyncio.TimeoutError:
                chunk = ""

            if chunk:
                buf += chunk

                if len(buf) > 20000:
                    buf = buf[-20000:]

                buf_lower = buf.lower()

                if any(p in buf_lower for p in _ACCEPT_A_PATTERNS):
                    logger.info(f"[{device_ip}] Detected accept banner, sending 'a'")
                    proc.stdin.write("a\n")
                    await proc.stdin.drain()
                    await asyncio.sleep(0.5)
                    buf = ""
                    consecutive_prompt_hits = 0
                    continue

                if any(p in buf_lower for p in _PRESS_ANY_KEY_PATTERNS):
                    logger.info(f"[{device_ip}] Detected press-any-key banner, sending ENTER")
                    proc.stdin.write("\n")
                    await proc.stdin.drain()
                    await asyncio.sleep(0.5)
                    buf = ""
                    consecutive_prompt_hits = 0
                    continue

                if _PROMPT_RE.search(buf):
                    consecutive_prompt_hits += 1
                    if consecutive_prompt_hits >= 2:
                        logger.info(f"[{device_ip}] Stable prompt detected")
                        break
                    await asyncio.sleep(0.3)
                    continue

                consecutive_prompt_hits = 0
            else:
                await asyncio.sleep(0.1)

    except Exception as ex:
        logger.warning(f"[{device_ip}] Banner handling failed: {ex}")
    finally:
        try:
            if proc:
                proc.close()
        except Exception:
            pass

    await asyncio.sleep(0.3)


async def _connect_device(
    bastion_conn: asyncssh.SSHClientConnection,
    ctx: dict[str, Any],
    connect_timeout: int,
) -> asyncssh.SSHClientConnection:
    conn = await asyncssh.connect(
        host=ctx["device_ip"],
        port=int(ctx["port"]),
        username=ctx["username"],
        password=ctx["password"],
        known_hosts=None,
        connect_timeout=connect_timeout,
        tunnel=bastion_conn,
    )

    logger.info(f"[{ctx['device_ip']}] SSH transport connected/authenticated")

    await _handle_interactive_login_banner(
        conn,
        ctx,
        timeout=min(12, max(6, connect_timeout)),
    )

    for attempt in range(2):
        try:
            await asyncio.wait_for(
                conn.run("terminal length 0", check=False),
                timeout=5,
            )
            break
        except Exception as ex:
            logger.warning(
                f"[{ctx['device_ip']}] terminal length 0 failed "
                f"attempt {attempt + 1}: {ex}"
            )
            if attempt == 0:
                await asyncio.sleep(1)

    return conn


async def _run_command(
    conn: asyncssh.SSHClientConnection,
    cmd: str,
    timeout: int = 60,
) -> str:
    cmd = cmd.strip()
    if not cmd:
        return ""

    r = await asyncio.wait_for(conn.run(cmd, check=False), timeout=timeout)
    output = (r.stdout or "") + ("\n" + r.stderr if r.stderr else "")

    lines = output.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    if lines and lines[0].strip() == cmd:
        lines = lines[1:]

    return "\n".join(lines)


async def _run_multiline_commands(
    conn: asyncssh.SSHClientConnection,
    commands: list[str],
    timeout: int = 60,
) -> str:
    outputs = []
    for cmd in commands:
        outputs.append(await _run_command(conn, cmd, timeout=timeout))
    return "\n".join(outputs)


async def _execute_checks(
    conn: asyncssh.SSHClientConnection,
    ctx: dict[str, Any],
    read_timeout: int = 60,
    max_retries: int = 3,
) -> dict[str, Any]:
    result_dict: dict[str, Any] = {}
    checks = ctx["checks"]["commands"] or {}
    brand_category = ctx["brand_category"]
    host_group = ctx.get("host_group")
    network_command_executer = NetworkFactory.get_network_brand_object(brand_category)

    for each_check, command in checks.items():
        cmd = command["command"]
        cmd_pattern = command["pattern"]
        check_overrides = None

        if host_group is not None:
            check_overrides = host_group.checks_override.get(each_check)
            if check_overrides is not None:
                cmd = check_overrides.get_command_str(cmd)

        commands_list = [c.strip() for c in cmd.split(",") if c.strip()]
        last_error = None

        for attempt in range(max_retries):
            try:
                if len(commands_list) == 1:
                    raw_output = await _run_command(
                        conn,
                        commands_list[0],
                        timeout=read_timeout,
                    )
                else:
                    raw_output = await _run_multiline_commands(
                        conn,
                        commands_list,
                        timeout=read_timeout,
                    )

                if "override_" in (brand_category or ""):
                    status, output = network_command_executer(
                        ctx["device_type"],
                        each_check,
                        raw_output,
                        cmd_pattern,
                        ctx["dashboard"],
                        brand_category,
                    )
                else:
                    if check_overrides is not None:
                        if check_overrides.parser_func_override is not None:
                            status, output = check_overrides.parser_func_override()
                        elif check_overrides.args_override is not None:
                            status, output = network_command_executer(
                                each_check,
                                raw_output,
                                cmd_pattern,
                                check_overrides.args_override,
                            )
                        else:
                            status, output = network_command_executer(
                                each_check,
                                raw_output,
                                cmd_pattern,
                            )
                    else:
                        status, output = network_command_executer(
                            each_check,
                            raw_output,
                            cmd_pattern,
                        )

                check_key = each_check if each_check != "Link Status" else "Interface Status"
                result_dict[check_key] = {
                    "status": status,
                    "output": str(output) if output is not None else None,
                    "err": None,
                }

                logger.info(f"[{ctx['device_ip']}] Check '{each_check}' -> {status}")
                break

            except asyncio.TimeoutError:
                last_error = f"Timeout after {read_timeout}s"
                logger.warning(
                    f"[{ctx['device_ip']}] Timeout on '{each_check}' "
                    f"attempt {attempt + 1}"
                )
            except EOFError:
                last_error = "EOFError: connection dropped"
                logger.warning(
                    f"[{ctx['device_ip']}] EOFError on '{each_check}' "
                    f"attempt {attempt + 1}"
                )
                break
            except Exception as ex:
                last_error = str(ex)
                logger.warning(
                    f"[{ctx['device_ip']}] Error on '{each_check}' "
                    f"attempt {attempt + 1}: {ex}"
                )
        else:
            check_key = each_check if each_check != "Link Status" else "Interface Status"
            result_dict[check_key] = {
                "status": HealthStatus.FAIL.value,
                "output": None,
                "err": last_error,
            }

    result_dict["Ping Status"] = {
        "status": "REACHABLE",
        "output": None,
        "err": None,
    }

    return result_dict


def _build_device_result(
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


def _build_unreachable_result(
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

    return _build_device_result(ctx, result_dict)


def _build_only_ping_result(ctx: dict[str, Any]) -> dict[str, Any]:
    return _build_device_result(
        ctx,
        {
            "Ping Status": {
                "status": "REACHABLE",
                "output": None,
                "err": None,
            }
        },
    )


async def _device_worker(
    ctx: dict[str, Any],
    bastion_conn: asyncssh.SSHClientConnection,
    semaphore: asyncio.Semaphore,
    connect_timeout: int,
    read_timeout: int,
) -> dict[str, Any]:
    async with semaphore:
        device_ip = ctx["device_ip"]
        device_name = ctx["device_name"]

        ping_ok = await _ping_via_bastion(bastion_conn, device_ip)
        if not ping_ok:
            logger.warning(f"[{device_ip}] Device not reachable, ping failed")
            return _build_unreachable_result(ctx, "Device not reachable")

        if ctx["only_ping_check"]:
            logger.info(f"[{device_ip}] Only ping check configured")
            return _build_only_ping_result(ctx)

        device_conn: asyncssh.SSHClientConnection | None = None

        try:
            device_conn = await _connect_device(
                bastion_conn,
                ctx,
                connect_timeout,
            )
            logger.info(f"[{device_ip}] SSH connected and ready")
        except asyncio.TimeoutError:
            logger.error(f"[{device_ip}] SSH connect timeout")
            return _build_unreachable_result(ctx, "SSH connection timeout")
        except asyncssh.DisconnectError as ex:
            logger.error(f"[{device_ip}] SSH disconnect: {ex}")
            return _build_unreachable_result(ctx, f"SSH disconnect: {ex}")
        except Exception as ex:
            logger.error(f"[{device_ip}] SSH connect error: {ex}\n{format_exc()}")
            return _build_unreachable_result(ctx, f"SSH error: {ex}")

        try:
            result_dict = await _execute_checks(
                device_conn,
                ctx,
                read_timeout=read_timeout,
            )
            logger.info(f"[{device_name}] All checks done")
            return _build_device_result(ctx, result_dict)
        except Exception as ex:
            logger.error(
                f"[{device_ip}] Unexpected error during checks: "
                f"{ex}\n{format_exc()}"
            )
            return _build_unreachable_result(ctx, f"Unexpected error: {ex}")
        finally:
            try:
                if device_conn:
                    device_conn.close()
                    await device_conn.wait_closed()
            except Exception:
                pass


async def run_framan_dashboard(
    dashboards: list[str],
    connect_timeout: int = 30,
    read_timeout: int = 60,
    device_concurrency: int = 3,
    device_names: Optional[list[str]] = None,
    device_ips: Optional[list[str]] = None,
    device_name_regex: Optional[str] = None,
    exclude_names: Optional[list[str]] = None,
    exclude_ips: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
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

    before = len(list_of_devices)
    list_of_devices = _filter_devices(
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

    logger.info(f"Loaded {len(list_of_devices)} devices for dashboards: {dashboards}")

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
            logger.warning(f"No region found for device {device.device_name}, skipping")
            continue

        ctx = _build_device_context(
            device,
            commands_dict,
            device_brand_checks_list,
            device_dashboard_override,
            device_brand_region,
        )

        region_device_map.setdefault(device_brand_region, []).append(ctx)

    all_results: list[dict[str, Any]] = []

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
                semaphore = asyncio.Semaphore(device_concurrency)

                tasks = [
                    _device_worker(
                        ctx,
                        bastion_conn,
                        semaphore,
                        connect_timeout,
                        read_timeout,
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
                    _build_unreachable_result(
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
    connect_timeout: int = 30,
    read_timeout: int = 60,
    device_concurrency: int = 3,
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
            device_names=device_names,
            device_ips=device_ips,
            device_name_regex=device_name_regex,
            exclude_names=exclude_names,
            exclude_ips=exclude_ips,
        )
    )


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)

    results = run_framan_dashboard_sync(
        dashboards=["dashboard_pcn"],
        connect_timeout=30,
        read_timeout=60,
        device_concurrency=3,
        device_names=[],
    )

    print(json.dumps(results, indent=2, default=str))
