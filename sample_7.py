import asyncio
import logging
import re
from typing import Any, Optional, Iterable
from traceback import format_exc
from types import SimpleNamespace

import asyncssh

from app.config import BaseConfig as Settings
from app.dal.nornir_dal import NornirDal
from app.utils.constants import HealthStatus, NETMIKO_TIMED_BRAND
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


# ---------------------------------------------------------------------------
# Helpers: device filtering
# ---------------------------------------------------------------------------

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
        if name_set and dn in name_set:
            return True
        if ip_set and dip in ip_set:
            return True
        if name_re and name_re.search(dn):
            return True
        return False

    def matches_exclude(d) -> bool:
        dn = getattr(d, "device_name", "") or ""
        dip = getattr(d, "device_ip", "") or ""
        return (dn in ex_name_set) or (dip in ex_ip_set)

    out: list[Any] = []
    for d in devices:
        if matches_include(d) and not matches_exclude(d):
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# Helpers: device context + platform
# ---------------------------------------------------------------------------

def _build_device_context(
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
        # If you want empty dict to behave as only-ping, change to:
        # only_ping_check = not commands
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


def _detect_platform(ctx: dict[str, Any]) -> str:
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


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------

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


_PROMPT_RE = re.compile(
    r"(?m)(?:^|\n).{0,120}(\s+\(.*\))?\s*[#>$]\s*$"
)

_ANSI_ESCAPE_RE = re.compile(
    r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])"
)

_CONTROL_CHARS_RE = re.compile(
    r"[\x00-\x07\x0b\x0c\x0e-\x1f\x7f]"
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
    "continue",
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
    """
    Handle interactive banners which block login, e.g. (Press 'a' to accept).

    This opens a temporary interactive shell, reads initial output,
    auto-sends responses, and waits until a stable prompt appears (best-effort).
    """
    device_ip = ctx.get("device_ip", "unknown")

    proc = None
    buf = ""
    start = asyncio.get_event_loop().time()

    try:
        proc = await conn.create_process(term_type="vt100")

        # Give the device a moment to send the initial banner.
        await asyncio.sleep(0.5)

        consecutive_prompt_hits = 0

        while True:
            elapsed = asyncio.get_event_loop().time() - start
            if elapsed > timeout:
                logger.warning(
                    f"[{device_ip}] Banner handling timeout after {timeout}s"
                )
                break

            try:
                chunk = await asyncio.wait_for(
                    proc.stdout.read(1024),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                chunk = ""

            if chunk:
                buf += chunk

                if len(buf) > 20000:
                    buf = buf[-20000:]

                buf_lower = buf.lower()

                if any(p in buf_lower for p in _ACCEPT_A_PATTERNS):
                    logger.info(
                        f"[{device_ip}] Detected accept banner, sending 'a'"
                    )
                    proc.stdin.write("a\n")
                    await proc.stdin.drain()
                    await asyncio.sleep(0.5)
                    buf = ""
                    consecutive_prompt_hits = 0
                    continue

                if any(p in buf_lower for p in _PRESS_ANY_KEY_PATTERNS):
                    logger.info(
                        f"[{device_ip}] Detected press-any-key banner, sending ENTER"
                    )
                    proc.stdin.write("\n")
                    await proc.stdin.drain()
                    await asyncio.sleep(0.5)
                    buf = ""
                    consecutive_prompt_hits = 0
                    continue

                if _PROMPT_RE.search(buf):
                    consecutive_prompt_hits += 1
                    if consecutive_prompt_hits >= 2:
                        logger.info(
                            f"[{device_ip}] Stable prompt detected, banner handling complete"
                        )
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


class _DeviceShellSession:
    """
    Keep one interactive CLI shell open for the whole device session.

    Many network devices do not behave like Linux SSH servers. They may reject
    SSH exec requests, or close the SSH transport after an exec-style command.
    This wrapper runs terminal setup and show commands through one persistent
    shell channel instead.
    """

    def __init__(
        self,
        conn: asyncssh.SSHClientConnection,
        ctx: dict[str, Any],
    ) -> None:
        self.conn = conn
        self.ctx = ctx
        self.proc = None

    def is_closed(self) -> bool:
        return self.conn.is_closed()

    async def open(self, timeout: int = 12) -> None:
        device_ip = self.ctx.get("device_ip", "unknown")
        try:
            self.proc = await self.conn.create_process(
                term_type="vt100",
                term_size=(511, 1000),
            )
        except TypeError:
            self.proc = await self.conn.create_process(term_type="vt100")
        await asyncio.sleep(0.5)
        await self._read_until_prompt(
            timeout=timeout,
            allow_banner_responses=True,
            send_initial_newline=True,
        )
        logger.info(f"[{device_ip}] Interactive shell ready")

    async def run(self, cmd: str, check: bool = False):
        del check
        stdout = await self.run_command(cmd)
        return SimpleNamespace(stdout=stdout, stderr="", exit_status=0)

    async def run_command(self, cmd: str, timeout: int = 60) -> str:
        if self.proc is None:
            raise ConnectionError("Interactive shell is not open")
        if self.conn.is_closed():
            raise ConnectionError("SSH connection is closed")

        cmd = cmd.strip()
        if not cmd:
            return ""

        await self._drain_available()

        self.proc.stdin.write(cmd + "\n")
        await self.proc.stdin.drain()

        output = await self._read_until_prompt(
            timeout=timeout,
            allow_banner_responses=False,
            send_initial_newline=False,
        )
        return self._clean_command_output(cmd, output)

    async def interrupt_current_command(self, timeout: int = 5) -> None:
        if self.proc is None:
            return
        if self.conn.is_closed():
            raise ConnectionError("SSH connection is closed")

        self.proc.stdin.write("\x03")
        await self.proc.stdin.drain()
        await self._read_until_prompt(
            timeout=timeout,
            allow_banner_responses=False,
            send_initial_newline=True,
        )
        await self._drain_available()

    async def _read_until_prompt(
        self,
        timeout: int,
        allow_banner_responses: bool,
        send_initial_newline: bool,
    ) -> str:
        if self.proc is None:
            raise ConnectionError("Interactive shell is not open")

        device_ip = self.ctx.get("device_ip", "unknown")
        buf = ""
        empty_reads = 0
        start = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start
            if elapsed > timeout:
                raise asyncio.TimeoutError(
                    f"Timed out waiting for prompt after {timeout}s"
                )

            if self.conn.is_closed():
                raise ConnectionError("SSH connection closed while waiting for prompt")

            try:
                chunk = await asyncio.wait_for(
                    self.proc.stdout.read(1024),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                chunk = ""

            if not chunk:
                empty_reads += 1
                if send_initial_newline and empty_reads == 2:
                    self.proc.stdin.write("\n")
                    await self.proc.stdin.drain()
                await asyncio.sleep(0.1)
                continue

            empty_reads = 0
            buf += chunk
            if len(buf) > 20000:
                buf = buf[-20000:]

            buf_lower = buf.lower()

            if allow_banner_responses and any(p in buf_lower for p in _ACCEPT_A_PATTERNS):
                logger.info(f"[{device_ip}] Detected accept banner, sending 'a'")
                self.proc.stdin.write("a\n")
                await self.proc.stdin.drain()
                await asyncio.sleep(0.5)
                buf = ""
                continue

            if allow_banner_responses and any(p in buf_lower for p in _PRESS_ANY_KEY_PATTERNS):
                logger.info(f"[{device_ip}] Detected press-any-key banner, sending ENTER")
                self.proc.stdin.write("\n")
                await self.proc.stdin.drain()
                await asyncio.sleep(0.5)
                buf = ""
                continue

            if self._buffer_has_prompt(buf):
                buf = await self._settle_after_prompt(buf)
                return buf

    async def _drain_available(self) -> None:
        if self.proc is None:
            return

        for _ in range(5):
            try:
                await asyncio.wait_for(self.proc.stdout.read(4096), timeout=0.05)
            except asyncio.TimeoutError:
                break

    async def _settle_after_prompt(self, buf: str) -> str:
        if self.proc is None:
            return buf

        await asyncio.sleep(0.2)
        for _ in range(5):
            try:
                chunk = await asyncio.wait_for(
                    self.proc.stdout.read(4096),
                    timeout=0.05,
                )
            except asyncio.TimeoutError:
                break
            if not chunk:
                break
            buf += chunk
        return buf

    @staticmethod
    def _buffer_has_prompt(buf: str) -> bool:
        clean = _DeviceShellSession._strip_terminal_control(buf)
        lines = clean.replace("\r\n", "\n").replace("\r", "\n").splitlines()
        for line in reversed(lines):
            if not line.strip():
                continue
            return bool(_PROMPT_RE.search("\n" + line))
        return False

    @staticmethod
    def _clean_command_output(cmd: str, output: str) -> str:
        output = _DeviceShellSession._strip_terminal_control(output)
        lines = output.replace("\r\n", "\n").replace("\r", "\n").splitlines()

        while lines and not lines[0].strip():
            lines.pop(0)

        while lines and _DeviceShellSession._is_echo_line(lines[0], cmd):
            lines = lines[1:]

        while lines and not lines[-1].strip():
            lines.pop()

        if lines and _PROMPT_RE.search("\n" + lines[-1]):
            lines = lines[:-1]

        return "\n".join(lines)

    @staticmethod
    def _strip_terminal_control(output: str) -> str:
        output = _ANSI_ESCAPE_RE.sub("", output)

        chars: list[str] = []
        for ch in output:
            if ch == "\b":
                if chars and chars[-1] not in "\r\n":
                    chars.pop()
                continue
            chars.append(ch)

        output = "".join(chars)
        return _CONTROL_CHARS_RE.sub("", output)

    @staticmethod
    def _is_echo_line(line: str, cmd: str) -> bool:
        line_norm = re.sub(r"\s+", " ", line.strip())
        cmd_norm = re.sub(r"\s+", " ", cmd.strip())

        if not line_norm:
            return True
        if line_norm == cmd_norm:
            return True
        if line_norm.startswith(cmd_norm[: min(len(cmd_norm), 30)]):
            return True
        if cmd_norm.startswith(line_norm) and len(line_norm) >= 8:
            return True
        return False

    def close(self) -> None:
        try:
            if self.proc is not None:
                self.proc.close()
        except Exception:
            pass
        self.conn.close()

    async def wait_closed(self) -> None:
        await self.conn.wait_closed()


async def _disable_paging(
    conn: Any,
    ctx: dict[str, Any],
    retries: int = 2,
) -> bool:
    """
    Disable paging if the device accepts Cisco-style terminal setup.

    Returns True when the command completed. Returns False when it timed out or
    failed while the SSH connection stayed open. Raises if the SSH connection
    itself closes, because the caller must reconnect in that case.
    """
    device_ip = ctx["device_ip"]

    try:
        await asyncio.wait_for(
            conn.run("terminal width 511", check=False),
            timeout=5,
        )
    except Exception as ex:
        logger.debug(f"[{device_ip}] terminal width 511 skipped/failed: {ex}")
        if conn.is_closed():
            raise ConnectionError(
                f"SSH connection closed during terminal width setup: {ex}"
            )

    for attempt in range(retries):
        try:
            await asyncio.wait_for(
                conn.run("terminal length 0", check=False),
                timeout=5,
            )
            return True
        except Exception as ex:
            logger.warning(
                f"[{device_ip}] terminal length 0 failed "
                f"attempt {attempt + 1}: {ex}"
            )
            if conn.is_closed():
                raise ConnectionError(
                    f"SSH connection closed during terminal setup: {ex}"
                )
            if attempt + 1 < retries:
                await asyncio.sleep(1)

    return False


async def _connect_device(
    bastion_conn: asyncssh.SSHClientConnection,
    ctx: dict[str, Any],
    connect_timeout: int,
) -> _DeviceShellSession:
    """Open an SSH tunnel to the device through the bastion and start an interactive CLI shell."""
    conn = await asyncssh.connect(
        host=ctx["device_ip"],
        port=int(ctx["port"]),
        username=ctx["username"],
        password=ctx["password"],
        known_hosts=None,
        connect_timeout=connect_timeout,
        tunnel=bastion_conn,
        # preferred_algs={ ... }  # enable only if you must for legacy gear
    )

    logger.info(f"[{ctx['device_ip']}] SSH authenticated")

    session = _DeviceShellSession(conn, ctx)
    await session.open(
        timeout=min(12, max(6, connect_timeout)),
    )

    await _disable_paging(session, ctx, retries=2)

    return session


async def _run_command(
    conn: Any,
    cmd: str,
    timeout: int = 60,
) -> str:
    """Run a single command and return stdout."""
    cmd = cmd.strip()
    if not cmd:
        return ""

    if hasattr(conn, "run_command"):
        return await conn.run_command(cmd, timeout=timeout)

    r = await asyncio.wait_for(conn.run(cmd, check=False), timeout=timeout)
    output = (r.stdout or "") + ("\n" + r.stderr if r.stderr else "")

    lines = output.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    if lines and lines[0].strip() == cmd.strip():
        lines = lines[1:]

    return "\n".join(lines)


async def _run_multiline_commands(
    conn: Any,
    commands: list[str],
    timeout: int = 60,
) -> str:
    """Run commands sequentially and concatenate output (non-interactive)."""
    outputs = []
    for cmd in commands:
        out = await _run_command(conn, cmd.strip(), timeout=timeout)
        outputs.append(out)
    return "\n".join(outputs)


def _unpack_parser_result(
    parser_result,
    ctx: dict[str, Any],
    check_name: str,
    raw_output: str,
) -> tuple[Any, Any]:
    if parser_result is None:
        preview = raw_output.replace("\n", "\\n")[:500]
        raise ValueError(
            f"Parser returned None for '{check_name}' on "
            f"{ctx['device_ip']} (raw_len={len(raw_output)}, "
            f"raw_preview={preview!r})"
        )

    try:
        status, output = parser_result
    except TypeError as ex:
        preview = raw_output.replace("\n", "\\n")[:500]
        raise ValueError(
            f"Parser returned invalid result for '{check_name}' on "
            f"{ctx['device_ip']}: {parser_result!r} "
            f"(raw_len={len(raw_output)}, raw_preview={preview!r})"
        ) from ex

    return status, output


def _get_check_int_override(
    overrides: Optional[dict[str, int]],
    check_name: str,
    default: int,
    minimum: int = 1,
) -> int:
    if not overrides:
        return default

    value = overrides.get(check_name)
    if value is None:
        check_name_lower = check_name.lower()
        for key, candidate in overrides.items():
            if key.lower() == check_name_lower:
                value = candidate
                break

    if value is None:
        return default

    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        logger.warning(
            f"Ignoring invalid override for check '{check_name}': {value!r}"
        )
        return default


# ---------------------------------------------------------------------------
# Per-device execution (mirrors execute_show_commands)
# ---------------------------------------------------------------------------

async def _execute_checks(
    conn: Any,
    ctx: dict[str, Any],
    read_timeout: int = 60,
    max_retries: int = 3,
    check_timeout_overrides: Optional[dict[str, int]] = None,
    check_retry_overrides: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    """
    Execute all checks for a device and return result dict.
    Mirrors DeviceHelper.execute_show_commands().
    """
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
        check_read_timeout = _get_check_int_override(
            check_timeout_overrides,
            each_check,
            read_timeout,
        )
        check_max_retries = _get_check_int_override(
            check_retry_overrides,
            each_check,
            max_retries,
        )

        for attempt in range(check_max_retries):
            try:
                if len(commands_list) == 1:
                    raw_output = await _run_command(
                        conn,
                        commands_list[0],
                        timeout=check_read_timeout,
                    )
                else:
                    raw_output = await _run_multiline_commands(
                        conn,
                        commands_list,
                        timeout=check_read_timeout,
                    )

                if "override_" in (brand_category or ""):
                    status, output = _unpack_parser_result(
                        network_command_executer(
                            ctx["device_type"],
                            each_check,
                            raw_output,
                            cmd_pattern,
                            ctx["dashboard"],
                            brand_category,
                        ),
                        ctx,
                        each_check,
                        raw_output,
                    )
                else:
                    if check_overrides is not None:
                        if check_overrides.parser_func_override is not None:
                            status, output = _unpack_parser_result(
                                check_overrides.parser_func_override(),
                                ctx,
                                each_check,
                                raw_output,
                            )
                        elif check_overrides.args_override is not None:
                            status, output = _unpack_parser_result(
                                network_command_executer(
                                    each_check,
                                    raw_output,
                                    cmd_pattern,
                                    check_overrides.args_override,
                                ),
                                ctx,
                                each_check,
                                raw_output,
                            )
                        else:
                            status, output = _unpack_parser_result(
                                network_command_executer(
                                    each_check,
                                    raw_output,
                                    cmd_pattern,
                                ),
                                ctx,
                                each_check,
                                raw_output,
                            )
                    else:
                        status, output = _unpack_parser_result(
                            network_command_executer(
                                each_check,
                                raw_output,
                                cmd_pattern,
                            ),
                            ctx,
                            each_check,
                            raw_output,
                        )

                check_key = each_check if each_check != "Link Status" else "Interface Status"
                result_dict[check_key] = {
                    "status": status,
                    "output": str(output) if output is not None else None,
                    "err": None,
                }
                logger.info(
                    f"[{ctx['device_ip']}] Check '{each_check}' -> {status}"
                )
                break

            except asyncio.TimeoutError:
                last_error = f"Timeout after {check_read_timeout}s"
                logger.warning(
                    f"[{ctx['device_ip']}] Timeout on '{each_check}' "
                    f"attempt {attempt + 1}/{check_max_retries}"
                )
                if hasattr(conn, "interrupt_current_command"):
                    try:
                        await conn.interrupt_current_command()
                    except Exception as recover_ex:
                        raise ConnectionError(
                            f"Unable to recover shell after timeout on "
                            f"'{each_check}': {recover_ex}"
                        ) from recover_ex
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


# ---------------------------------------------------------------------------
# Result builders
# ---------------------------------------------------------------------------

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
        {"Ping Status": {"status": "REACHABLE", "output": None, "err": None}},
    )


# ---------------------------------------------------------------------------
# Per-device worker
# ---------------------------------------------------------------------------

async def _device_worker(
    ctx: dict[str, Any],
    bastion_conn: asyncssh.SSHClientConnection,
    semaphore: asyncio.Semaphore,
    connect_timeout: int,
    read_timeout: int,
    ping_timeout: int = 5,
    ping_before_connect: bool = True,
    check_timeout_overrides: Optional[dict[str, int]] = None,
    check_retry_overrides: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    async with semaphore:
        device_ip = ctx["device_ip"]
        device_name = ctx["device_name"]

        # 1) Ping check via bastion.
        if ping_before_connect or ctx["only_ping_check"]:
            ping_ok = await _ping_via_bastion(
                bastion_conn,
                device_ip,
                timeout=ping_timeout,
            )
            if not ping_ok:
                logger.warning(f"[{device_ip}] Device not reachable (ping failed)")
                return _build_unreachable_result(ctx, "Device not reachable")

        # 2) Only-ping devices.
        if ctx["only_ping_check"]:
            logger.info(f"[{device_ip}] Only ping check configured")
            return _build_only_ping_result(ctx)

        # 3) SSH connect via bastion tunnel.
        device_conn: Any | None = None
        try:
            # Important: do not wrap _connect_device() in wait_for(connect_timeout).
            # _connect_device() includes SSH auth, banner handling, and terminal setup.
            # Wrapping the whole flow can create false "SSH connection timeout" errors.
            device_conn = await _connect_device(
                bastion_conn,
                ctx,
                connect_timeout,
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

        # 4) Execute checks.
        try:
            result_dict = await _execute_checks(
                device_conn,
                ctx,
                read_timeout=read_timeout,
                check_timeout_overrides=check_timeout_overrides,
                check_retry_overrides=check_retry_overrides,
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
                if device_conn is not None:
                    device_conn.close()
                    await device_conn.wait_closed()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

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
    commands_dict = get_brand_models_commands(brand_model_with_checks, brand_model_config)
    list_of_devices = NornirDal.get_all_devices(dashboards)

    # Optional filtering.
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

        ctx = _build_device_context(
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
                    _device_worker(
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


# ---------------------------------------------------------------------------
# Local runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

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
