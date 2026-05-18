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

# ─────────────────────────────────────────────
# Prompt + banner pattern constants
# ─────────────────────────────────────────────

_PROMPT_RE = re.compile(
    r"(?m)(?:^|\n).{0,120}"
    r"(\s+\(.*\)|\[.*\])?"   # (config) or [edit] mode suffixes
    r"\s*[#>$%]\s*$"          # ending char — covers most NOS prompts
)

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


# ─────────────────────────────────────────────
# Helpers: device filtering
# ─────────────────────────────────────────────

def _filter_devices(
    devices: list[Any],
    device_names: Optional[Iterable[str]] = None,
    device_ips: Optional[Iterable[str]] = None,
    device_name_regex: Optional[str] = None,
    exclude_names: Optional[Iterable[str]] = None,
    exclude_ips: Optional[Iterable[str]] = None,
) -> list[Any]:
    """
    Filter device objects from DB (must have device_name and device_ip attributes).

    Include logic:
      - If any include filter is provided (names/ips/regex): device must match ANY of them.
      - If no include filters are provided: include all.

    Exclude logic:
      - Excludes are applied last (always).
    """
    name_set    = {str(x).strip() for x in (device_names or []) if x and str(x).strip()}
    ip_set      = {str(x).strip() for x in (device_ips or []) if x and str(x).strip()}
    ex_name_set = {str(x).strip() for x in (exclude_names or []) if x and str(x).strip()}
    ex_ip_set   = {str(x).strip() for x in (exclude_ips or []) if x and str(x).strip()}

    name_re = re.compile(device_name_regex) if device_name_regex else None

    def matches_include(d) -> bool:
        if not name_set and not ip_set and not name_re:
            return True
        dn  = getattr(d, "device_name", "") or ""
        dip = getattr(d, "device_ip", "") or ""
        if name_set and dn in name_set:
            return True
        if ip_set and dip in ip_set:
            return True
        if name_re and name_re.search(dn):
            return True
        return False

    def matches_exclude(d) -> bool:
        dn  = getattr(d, "device_name", "") or ""
        dip = getattr(d, "device_ip", "") or ""
        return (dn in ex_name_set) or (dip in ex_ip_set)

    out: list[Any] = []
    for d in devices:
        if matches_include(d) and not matches_exclude(d):
            out.append(d)
    return out


# ─────────────────────────────────────────────
# Helpers: device context + platform
# ─────────────────────────────────────────────

def _build_device_context(
    device,
    commands_dict: dict,
    device_brand_checks_list,
    device_dashboard_ovveride: dict,
    device_brand_region: str,
) -> dict[str, Any]:
    """
    Mirror the logic of get_device_host_object() from nornir_inventory_manager.
    Returns a plain dict instead of a Nornir Host object.
    """
    brand_model   = device.brand_model
    device_type   = device.device_type
    name          = device.device_name
    dashboard     = device.dashboard

    only_ping_check  = True
    brand_category   = None
    commands         = None
    username         = None
    password         = None

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
            commands       = override_brand
            brand_category = "override_" + brand_model
        else:
            commands = get_all_check_for_device(
                commands_dict[brand_model]["checks"], device_brand_checks_list
            )
            brand_category = commands_dict[brand_model]["brand_category"]

        override_region = get_region_override_data().get(dashboard)
        if override_region:
            commands, brand_category = get_region_override_commands(
                commands, dashboard, name, brand_model, brand_category
            )

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
        "device_name":    name,
        "device_ip":      device.device_ip,
        "port":           device.port,
        "dashboard":      dashboard,
        "assert_id":      device.assert_id,
        "infra_type":     device.infra_type,
        "infra_name":     device.infra_name,
        "brand_model":    brand_model,
        "device_id":      device.id,
        "device_type":    device_type,
        "os_type":        device.os_type,
        "brand_category": brand_category,
        "only_ping_check": only_ping_check,
        "region":         device_brand_region,
        "username":       username,
        "password":       password,
        "host_group":     host_group,
        "checks":         {"commands": commands},
    }


def _detect_platform(ctx: dict[str, Any]) -> str:
    """Mirror fetch_host_platform() logic from device_helper."""
    from app.utils.constants import (
        ARISTA_SERIES, CHECK_POINT_SERIES, CITRIX_SERIES,
        F5_NETWORK_SERIES, SKYHIGH_SERIES, FORTINET_SERIES,
    )

    brand_category = ctx.get("brand_category", "")
    os_type        = ctx.get("os_type", "")

    if brand_category in FORTINET_SERIES:
        return "fortinet"
    elif brand_category in ARISTA_SERIES:
        return "cisco_ios" if os_type == "mos" else "arista_eos"
    elif brand_category in CITRIX_SERIES:
        return "netscaler"
    elif brand_category in CHECK_POINT_SERIES:
        return "cisco_ios"
    elif brand_category in F5_NETWORK_SERIES:
        return "f5_tmsh"
    elif brand_category in SKYHIGH_SERIES:
        return "linux"
    else:
        if os_type == "nx-os":
            return "cisco_nxos"
        elif os_type == "ios-xe":
            return "cisco_xe"
        elif os_type in ("ios-xrv", "ios-xr"):
            return "cisco_xr"
        return "cisco_ios"


# ─────────────────────────────────────────────
# SSH helpers
# ─────────────────────────────────────────────

async def _ping_via_bastion(
    bastion_conn: asyncssh.SSHClientConnection, ip: str, timeout: int = 5
) -> bool:
    try:
        r = await asyncio.wait_for(
            bastion_conn.run(f"ping -c 1 -W 1 {ip}", check=False),
            timeout=timeout,
        )
        return r.exit_status == 0
    except Exception:
        return False


async def _handle_banner_in_stream(
    proc,
    buf: str,
    device_ip: str,
) -> tuple[str, bool]:
    """
    Check buf for known interactive banners and respond.

    Returns:
        (buf, responded) — buf is cleared if a banner was handled,
        responded=True means a keypress was sent.
    """
    buf_lower = buf.lower()

    # ── Press 'a' to accept ─────────────────────────────────────────────
    if any(p in buf_lower for p in _ACCEPT_A_PATTERNS):
        logger.info(f"[{device_ip}] Banner detected → sending 'a'")
        proc.stdin.write("a\n")
        await proc.stdin.drain()
        await asyncio.sleep(0.5)
        return "", True  # clear buf, responded

    # ── --More-- / press any key paging ─────────────────────────────────
    if any(p in buf_lower for p in _PRESS_ANY_KEY_PATTERNS):
        logger.info(f"[{device_ip}] Paging prompt detected → sending SPACE")
        proc.stdin.write(" ")
        await proc.stdin.drain()
        await asyncio.sleep(0.3)
        return buf, True

    return buf, False  # nothing handled


async def _read_until_prompt(
    proc,
    device_ip: str,
    cmd: str,
    timeout: int,
    start: float,
) -> str:
    """
    Read stdout from proc until a stable shell prompt is detected
    or timeout is reached. Handles banners inline.

    Returns the raw output string for this command.
    """
    buf = ""
    consecutive_prompt_hits = 0

    while True:
        elapsed = asyncio.get_event_loop().time() - start
        if elapsed > timeout:
            logger.warning(
                f"[{device_ip}] Command '{cmd}' timed out after {timeout}s"
            )
            break

        try:
            chunk = await asyncio.wait_for(
                proc.stdout.read(4096), timeout=2.0
            )
        except asyncio.TimeoutError:
            if _PROMPT_RE.search(buf):
                break
            continue

        if not chunk:
            break

        buf += chunk

        if len(buf) > 50000:
            buf = buf[-50000:]

        buf, responded = await _handle_banner_in_stream(proc, buf, device_ip)
        if responded:
            consecutive_prompt_hits = 0
            # KEY FIX: After responding to a banner, wait for the device
            # to settle and emit a fresh stable prompt before continuing.
            # Without this, the next command races with the device still
            # processing the banner response.
            await asyncio.sleep(0.8)
            buf = ""  # discard banner noise, wait for clean prompt
            continue

        if _PROMPT_RE.search(buf):
            consecutive_prompt_hits += 1
            if consecutive_prompt_hits >= 2:
                logger.debug(f"[{device_ip}] Stable prompt after '{cmd}'")
                break
            await asyncio.sleep(0.3)
        else:
            consecutive_prompt_hits = 0

    return buf


def _clean_output(raw: str, cmd: str) -> str:
    """
    Strip echoed command (first line) and trailing prompt line from raw output.
    """
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").splitlines()

    if lines and lines[0].strip() == cmd.strip():
        lines = lines[1:]

    if lines and _PROMPT_RE.search(lines[-1]):
        lines = lines[:-1]

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Login banner handler (called once at connect)
# ─────────────────────────────────────────────

async def _handle_interactive_login_banner(
    conn: asyncssh.SSHClientConnection,
    ctx: dict[str, Any],
    timeout: int = 12,
) -> None:
    """
    Handle interactive banners at login time (e.g. Press 'a' to accept).
    Opens a temporary shell, waits for a stable prompt, then closes it.
    """
    device_ip = ctx.get("device_ip", "unknown")
    proc = None
    start = asyncio.get_event_loop().time()

    try:
        proc = await conn.create_process(term_type="vt100")
        await asyncio.sleep(0.5)

        buf = ""
        consecutive_prompt_hits = 0

        while True:
            elapsed = asyncio.get_event_loop().time() - start
            if elapsed > timeout:
                logger.warning(
                    f"[{device_ip}] Login banner handling timeout after {timeout}s"
                )
                break

            try:
                chunk = await asyncio.wait_for(
                    proc.stdout.read(1024), timeout=1.0
                )
            except asyncio.TimeoutError:
                chunk = ""

            if chunk:
                buf += chunk
                if len(buf) > 20000:
                    buf = buf[-20000:]

                buf, responded = await _handle_banner_in_stream(
                    proc, buf, device_ip
                )
                if responded:
                    consecutive_prompt_hits = 0
                    continue

                if _PROMPT_RE.search(buf):
                    consecutive_prompt_hits += 1
                    if consecutive_prompt_hits >= 2:
                        logger.info(
                            f"[{device_ip}] Login banner done — stable prompt"
                        )
                        break
                    await asyncio.sleep(0.3)
                    continue
                else:
                    consecutive_prompt_hits = 0
            else:
                await asyncio.sleep(0.1)

    except Exception as ex:
        logger.warning(f"[{device_ip}] Login banner handling failed: {ex}")
    finally:
        try:
            if proc:
                proc.close()
        except Exception:
            pass

    await asyncio.sleep(0.3)


# ─────────────────────────────────────────────
# Device connect
# ─────────────────────────────────────────────

async def _connect_device(
    bastion_conn: asyncssh.SSHClientConnection,
    ctx: dict[str, Any],
    connect_timeout: int,
) -> asyncssh.SSHClientConnection:
    """
    Open SSH tunnel to device through bastion and handle login banner.
    """
    conn = await asyncssh.connect(
        host=ctx["device_ip"],
        port=int(ctx["port"]),
        username=ctx["username"],
        password=ctx["password"],
        known_hosts=None,
        connect_timeout=connect_timeout,
        tunnel=bastion_conn,
    )

    await _handle_interactive_login_banner(
        conn, ctx, timeout=min(12, max(6, connect_timeout))
    )

    for attempt in range(2):
        try:
            await asyncio.wait_for(
                conn.run("terminal length 0", check=False), timeout=5
            )
            break
        except Exception:
            if attempt == 0:
                await asyncio.sleep(1)

    return conn


# ─────────────────────────────────────────────
# Per-device execution
# ─────────────────────────────────────────────

async def _execute_checks(
    conn: asyncssh.SSHClientConnection,
    ctx: dict[str, Any],
    read_timeout: int = 60,
    max_retries: int = 3,
) -> dict[str, Any]:
    """
    Execute all checks for a device using a SINGLE persistent shell session.

    Key improvement over previous version:
      - One `create_process` shell is opened for the ENTIRE device session.
      - Prefix/mode-setting commands (e.g. "conf global") are tracked in
        `already_run_prefixes` and only executed ONCE, no matter how many
        checks share that prefix.
      - This means "conf global" → POST WARNING banner only fires once,
        not once per check.

    Example DB commands:
      Check 1: "conf global,get system performance status | grep CPU"
      Check 2: "conf global,get system performance status | grep Memory"
      Check 3: "conf global,get system performance status | grep Uptime"

    Execution:
      → conf global  (runs once, banner handled once)
      → grep CPU     (output collected)
      → grep Memory  (output collected, conf global skipped)
      → grep Uptime  (output collected, conf global skipped)
    """
    result_dict: dict[str, Any] = {}
    checks                   = ctx["checks"]["commands"] or {}
    brand_category           = ctx["brand_category"]
    device_ip                = ctx["device_ip"]
    host_group               = ctx.get("host_group")
    network_command_executer = NetworkFactory.get_network_brand_object(brand_category)

    # Shared timeout budget across all checks for this device
    start = asyncio.get_event_loop().time()

    # ── Open ONE persistent shell for the entire device session ──────────
    proc = await conn.create_process(term_type="vt100")
    await asyncio.sleep(0.2)  # let shell settle

    # Tracks prefix commands already run in this shell so they are not
    # repeated for subsequent checks (e.g. "conf global" runs only once).
    already_run_prefixes: set[str] = set()

    try:
        for each_check, command in checks.items():
            cmd             = command["command"]
            cmd_pattern     = command["pattern"]
            check_overrides = None

            if host_group is not None:
                check_overrides = host_group.checks_override.get(each_check)
                if check_overrides is not None:
                    cmd = check_overrides.get_command_str(cmd)

            # Split comma-separated commands and strip whitespace
            # e.g. "conf global,get system performance status | grep CPU"
            # → ["conf global", "get system performance status | grep CPU"]
            commands_list = [c.strip() for c in cmd.split(",") if c.strip()]
            last_error    = None

            for attempt in range(max_retries):
                try:
                    elapsed   = asyncio.get_event_loop().time() - start
                    remaining = read_timeout - elapsed
                    if remaining <= 0:
                        raise asyncio.TimeoutError()

                    if len(commands_list) == 1:
                        # ── Single command: run directly in persistent shell ──
                        single_cmd = commands_list[0]
                        logger.debug(f"[{device_ip}] Sending: '{single_cmd}'")
                        proc.stdin.write(single_cmd + "\n")
                        await proc.stdin.drain()

                        raw = await _read_until_prompt(
                            proc, device_ip, single_cmd, int(remaining), start
                        )
                        raw_output = _clean_output(raw, single_cmd)

                    else:
                        # ── Multiline: reuse persistent shell, skip already-run prefixes ──
                        #
                        # Everything except the LAST command is treated as a
                        # "prefix" (mode-setter). The last command is the actual
                        # data command whose output we collect.
                        #
                        # already_run_prefixes ensures "conf global" only fires
                        # once across ALL checks in this device session.
                        raw_parts: list[str] = []

                        for i, c in enumerate(commands_list):
                            is_last   = (i == len(commands_list) - 1)
                            is_prefix = not is_last

                            if is_prefix and c in already_run_prefixes:
                                # Shell is already in this mode — skip entirely
                                logger.debug(
                                    f"[{device_ip}] Skipping already-run prefix: '{c}'"
                                )
                                continue

                            elapsed   = asyncio.get_event_loop().time() - start
                            remaining = read_timeout - elapsed
                            if remaining <= 0:
                                raise asyncio.TimeoutError()

                            logger.debug(f"[{device_ip}] Sending: '{c}'")
                            proc.stdin.write(c + "\n")
                            await proc.stdin.drain()

                            raw     = await _read_until_prompt(
                                proc, device_ip, c, int(remaining), start
                            )
                            cleaned = _clean_output(raw, c)

                            if is_prefix:
                                # Mark prefix as done so future checks skip it
                                already_run_prefixes.add(c)
                                logger.debug(
                                    f"[{device_ip}] Mode set via '{c}' — won't repeat"
                                )
                                # Prefix output is not collected (not useful data)
                            else:
                                # Only the last (data) command output is collected
                                raw_parts.append(cleaned)

                        raw_output = "\n".join(raw_parts)

                    # ── Parse output using NetworkFactory ────────────────────
                    if "override_" in (brand_category or ""):
                        status, output = network_command_executer(
                            ctx["device_type"], each_check, raw_output,
                            cmd_pattern, ctx["dashboard"], brand_category,
                        )
                    else:
                        if check_overrides is not None:
                            if check_overrides.parser_func_override is not None:
                                status, output = check_overrides.parser_func_override()
                            elif check_overrides.args_override is not None:
                                status, output = network_command_executer(
                                    each_check, raw_output, cmd_pattern,
                                    check_overrides.args_override
                                )
                            else:
                                status, output = network_command_executer(
                                    each_check, raw_output, cmd_pattern
                                )
                        else:
                            status, output = network_command_executer(
                                each_check, raw_output, cmd_pattern
                            )

                    check_key = (
                        each_check if each_check != "Link Status"
                        else "Interface Status"
                    )
                    result_dict[check_key] = {
                        "status": status,
                        "output": str(output) if output is not None else None,
                        "err":    None,
                    }
                    logger.info(f"[{device_ip}] Check '{each_check}' → {status}")
                    break  # success — move to next check

                except asyncio.TimeoutError:
                    last_error = f"Timeout after {read_timeout}s"
                    logger.warning(
                        f"[{device_ip}] Timeout on '{each_check}' "
                        f"attempt {attempt + 1}/{max_retries}"
                    )

                except EOFError:
                    last_error = "EOFError: connection dropped"
                    logger.warning(
                        f"[{device_ip}] EOFError on '{each_check}' "
                        f"attempt {attempt + 1}/{max_retries}"
                    )
                    check_key = (
                        each_check if each_check != "Link Status"
                        else "Interface Status"
                    )
                    result_dict[check_key] = {
                        "status": HealthStatus.FAIL.value,
                        "output": None,
                        "err":    last_error,
                    }
                    break  # connection dropped — no point retrying

                except Exception as ex:
                    last_error = str(ex)
                    logger.warning(
                        f"[{device_ip}] Error on '{each_check}' "
                        f"attempt {attempt + 1}/{max_retries}: {ex}"
                    )

            else:
                # All retries exhausted
                check_key = (
                    each_check if each_check != "Link Status"
                    else "Interface Status"
                )
                result_dict[check_key] = {
                    "status": HealthStatus.FAIL.value,
                    "output": None,
                    "err":    last_error,
                }

    finally:
        try:
            proc.close()
        except Exception:
            pass

    result_dict["Ping Status"] = {
        "status": "REACHABLE",
        "output": None,
        "err":    None,
    }
    return result_dict


# ─────────────────────────────────────────────
# Result builders
# ─────────────────────────────────────────────

def _build_device_result(
    ctx: dict[str, Any],
    result_dict: dict[str, Any],
) -> dict[str, Any]:
    return {
        "device_name":      ctx["device_name"],
        "device_ip":        ctx["device_ip"],
        "assert_id":        ctx.get("assert_id"),
        "dashboard":        ctx.get("dashboard"),
        "port":             ctx["port"],
        "brand_model":      ctx.get("brand_model"),
        "infra_name":       ctx.get("infra_name"),
        "infra_type":       ctx.get("infra_type"),
        "device_json_data": {ctx["device_name"]: result_dict},
    }


def _build_unreachable_result(
    ctx: dict[str, Any], reason: str
) -> dict[str, Any]:
    checks      = ctx["checks"]["commands"] or {}
    result_dict: dict[str, Any] = {}

    for each_check in checks:
        result_dict[each_check] = {
            "status": HealthStatus.NOTCONNECTED.value,
            "output": None,
            "err":    reason,
        }

    result_dict["Ping Status"] = {
        "status": "NOTREACHABLE",
        "output": None,
        "err":    reason,
    }
    return _build_device_result(ctx, result_dict)


def _build_only_ping_result(ctx: dict[str, Any]) -> dict[str, Any]:
    return _build_device_result(
        ctx,
        {"Ping Status": {"status": "REACHABLE", "output": None, "err": None}},
    )


# ─────────────────────────────────────────────
# Per-device worker
# ─────────────────────────────────────────────

async def _device_worker(
    ctx: dict[str, Any],
    bastion_conn: asyncssh.SSHClientConnection,
    semaphore: asyncio.Semaphore,
    connect_timeout: int,
    read_timeout: int,
) -> dict[str, Any]:
    async with semaphore:
        device_ip   = ctx["device_ip"]
        device_name = ctx["device_name"]

        # 1) Ping check via bastion
        ping_ok = await _ping_via_bastion(bastion_conn, device_ip)
        if not ping_ok:
            logger.warning(f"[{device_ip}] Device not reachable (ping failed)")
            return _build_unreachable_result(ctx, "Device not reachable")

        # 2) Only-ping devices
        if ctx["only_ping_check"]:
            logger.info(f"[{device_ip}] Only ping check configured")
            return _build_only_ping_result(ctx)

        # 3) SSH connect via bastion tunnel
        device_conn: asyncssh.SSHClientConnection | None = None
        try:
            device_conn = await asyncio.wait_for(
                _connect_device(bastion_conn, ctx, connect_timeout),
                timeout=connect_timeout,
            )
            logger.info(f"[{device_ip}] SSH connected")
        except asyncio.TimeoutError:
            logger.error(f"[{device_ip}] SSH connect timeout")
            return _build_unreachable_result(ctx, "SSH connection timeout")
        except asyncssh.DisconnectError as ex:
            logger.error(f"[{device_ip}] SSH disconnect: {ex}")
            return _build_unreachable_result(ctx, f"SSH disconnect: {ex}")
        except Exception as ex:
            logger.error(
                f"[{device_ip}] SSH connect error: {ex}\n{format_exc()}"
            )
            return _build_unreachable_result(ctx, f"SSH error: {ex}")

        # 4) Execute all checks (single persistent shell)
        try:
            result_dict = await _execute_checks(
                device_conn, ctx, read_timeout=read_timeout
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
                device_conn.close()
                await device_conn.wait_closed()
            except Exception:
                pass


# ─────────────────────────────────────────────
# Public entry-point
# ─────────────────────────────────────────────

async def run_framan_dashboard(
    dashboards: list[str],
    connect_timeout: int = 15,
    read_timeout: int = 60,
    device_concurrency: int = 20,
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
      4. Execute show commands in a single persistent shell per device
         (prefix commands like 'conf global' run only once per device)
      5. Return list of device payloads
    """

    # Load inventory/config
    device_dashboard_config   = get_device_dashboard_config()
    brand_model_with_checks   = remove_pinged_check_brand_model(device_dashboard_config)
    device_brand_config       = NornirDal.get_device_barand_model_config()
    device_dashboard_override = get_device_override_config()
    brand_model_config        = get_brand_model_pattern_config(device_brand_config)
    commands_dict             = get_brand_models_commands(brand_model_with_checks, brand_model_config)
    list_of_devices           = NornirDal.get_all_devices(dashboards)

    # Optional filtering
    before          = len(list_of_devices)
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
            f"Device filter applied: {before} → {after} devices "
            f"(names={device_names}, ips={device_ips}, "
            f"regex={device_name_regex}, "
            f"exclude_names={exclude_names}, exclude_ips={exclude_ips})"
        )

    logger.info(
        f"Loaded {len(list_of_devices)} devices for dashboards: {dashboards}"
    )

    # Group devices by region
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

        ctx = _build_device_context(
            device,
            commands_dict,
            device_brand_checks_list,
            device_dashboard_override,
            device_brand_region,
        )
        region_device_map.setdefault(device_brand_region, []).append(ctx)

    all_results: list[dict[str, Any]] = []

    # Process each region (one bastion connection per region)
    for region, device_contexts in region_device_map.items():
        bastion_host = Settings.DEVICE_CREDENTIALS[region]["JUMPHOST_IP"]
        bastion_user = Settings.DEVICE_CREDENTIALS[region]["JUMPHOST_USER"]
        bastion_port = int(Settings.DEVICE_CREDENTIALS[region].get("JUMPHOST_PORT", 22))
        bastion_key  = (
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
                        ctx, bastion_conn, semaphore,
                        connect_timeout, read_timeout
                    )
                    for ctx in device_contexts
                ]

                region_results = await asyncio.gather(
                    *tasks, return_exceptions=False
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
                        ctx, f"Jumphost not reachable: {ex}"
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
    device_concurrency: int = 20,
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
            device_names=device_names,
            device_ips=device_ips,
            device_name_regex=device_name_regex,
            exclude_names=exclude_names,
            exclude_ips=exclude_ips,
        )
    )


# ─────────────────────────────────────────────
# Local runner
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)

    results = run_framan_dashboard_sync(
        dashboards=["dashboard_pcn"],
        connect_timeout=15,
        read_timeout=60,
        device_concurrency=10,
        device_names=[],
    )

    print(json.dumps(results, indent=2, default=str))
