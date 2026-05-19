import asyncio
import logging
import re
from types import SimpleNamespace
from typing import Any

import asyncssh

logger = logging.getLogger(__name__)

PROMPT_RE = re.compile(
    r"(?m)(?:^|\n).{0,120}(\s+\(.*\))?\s*[#>$]\s*$"
)

ANSI_ESCAPE_RE = re.compile(
    r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])"
)

CONTROL_CHARS_RE = re.compile(
    r"[\x00-\x07\x0b\x0c\x0e-\x1f\x7f]"
)

ACCEPT_A_PATTERNS = [
    "press 'a' to accept",
    "press a to accept",
    "(press 'a' to accept)",
    "(press a to accept)",
]

PRESS_ANY_KEY_PATTERNS = [
    "press any key",
    "press enter",
    "hit any key",
    "continue",
    "press return",
    "--more--",
    "-- more --",
    "space for more",
]


async def ping_via_bastion(
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


class DeviceShellSession:
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

            if allow_banner_responses and any(p in buf_lower for p in ACCEPT_A_PATTERNS):
                logger.info(f"[{device_ip}] Detected accept banner, sending 'a'")
                self.proc.stdin.write("a\n")
                await self.proc.stdin.drain()
                await asyncio.sleep(0.5)
                buf = ""
                continue

            if allow_banner_responses and any(p in buf_lower for p in PRESS_ANY_KEY_PATTERNS):
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
        clean = DeviceShellSession._strip_terminal_control(buf)
        lines = clean.replace("\r\n", "\n").replace("\r", "\n").splitlines()
        for line in reversed(lines):
            if not line.strip():
                continue
            return bool(PROMPT_RE.search("\n" + line))
        return False

    @staticmethod
    def _clean_command_output(cmd: str, output: str) -> str:
        output = DeviceShellSession._strip_terminal_control(output)
        lines = output.replace("\r\n", "\n").replace("\r", "\n").splitlines()

        while lines and not lines[0].strip():
            lines.pop(0)

        while lines and DeviceShellSession._is_echo_line(lines[0], cmd):
            lines = lines[1:]

        while lines and not lines[-1].strip():
            lines.pop()

        if lines and PROMPT_RE.search("\n" + lines[-1]):
            lines = lines[:-1]

        return "\n".join(lines)

    @staticmethod
    def _strip_terminal_control(output: str) -> str:
        output = ANSI_ESCAPE_RE.sub("", output)

        chars: list[str] = []
        for ch in output:
            if ch == "\b":
                if chars and chars[-1] not in "\r\n":
                    chars.pop()
                continue
            chars.append(ch)

        output = "".join(chars)
        return CONTROL_CHARS_RE.sub("", output)

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


async def disable_paging(
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


async def connect_device(
    bastion_conn: asyncssh.SSHClientConnection,
    ctx: dict[str, Any],
    connect_timeout: int,
) -> DeviceShellSession:
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

    session = DeviceShellSession(conn, ctx)
    await session.open(
        timeout=min(12, max(6, connect_timeout)),
    )

    await disable_paging(session, ctx, retries=2)

    return session


async def run_command(
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


async def run_multiline_commands(
    conn: Any,
    commands: list[str],
    timeout: int = 60,
) -> str:
    """Run commands sequentially and concatenate output."""
    outputs = []
    for cmd in commands:
        out = await run_command(conn, cmd.strip(), timeout=timeout)
        outputs.append(out)
    return "\n".join(outputs)
