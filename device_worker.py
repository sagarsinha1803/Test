import asyncio
import logging
from traceback import format_exc
from typing import Any, Optional

import asyncssh

from .check_executor import execute_checks
from .result_builder import (
    build_device_result,
    build_only_ping_result,
    build_unreachable_result,
)
from .ssh_client import connect_device, ping_via_bastion

logger = logging.getLogger(__name__)


async def _connect_device_with_retry(
    ctx: dict[str, Any],
    bastion_conn: asyncssh.SSHClientConnection,
    connect_timeout: int,
    max_attempts: int = 2,
    retry_delay: float = 1.5,
) -> Any:
    device_ip = ctx["device_ip"]
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await connect_device(
                bastion_conn,
                ctx,
                connect_timeout,
            )
        except (asyncio.TimeoutError, asyncssh.DisconnectError, ConnectionError) as ex:
            last_error = ex
            if attempt >= max_attempts:
                raise

            logger.warning(
                f"[{device_ip}] SSH setup failed attempt "
                f"{attempt}/{max_attempts}: {ex}; retrying"
            )
            await asyncio.sleep(retry_delay)

    if last_error is not None:
        raise last_error
    raise ConnectionError("SSH setup failed without an exception")


async def device_worker(
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
            ping_ok = await ping_via_bastion(
                bastion_conn,
                device_ip,
                timeout=ping_timeout,
            )
            if not ping_ok:
                logger.warning(f"[{device_ip}] Device not reachable (ping failed)")
                return build_unreachable_result(ctx, "Device not reachable")

        # 2) Only-ping devices.
        if ctx["only_ping_check"]:
            logger.info(f"[{device_ip}] Only ping check configured")
            return build_only_ping_result(ctx)

        # 3) SSH connect via bastion tunnel.
        device_conn: Any | None = None
        try:
            device_conn = await _connect_device_with_retry(
                ctx,
                bastion_conn,
                connect_timeout,
            )
            logger.info(f"[{device_ip}] SSH connected")
        except asyncio.TimeoutError:
            logger.error(f"[{device_ip}] SSH connect timeout")
            return build_unreachable_result(ctx, "SSH connection timeout")
        except asyncssh.DisconnectError as ex:
            logger.error(f"[{device_ip}] SSH disconnect: {ex}")
            return build_unreachable_result(ctx, f"SSH disconnect: {ex}")
        except Exception as ex:
            logger.error(
                f"[{device_ip}] SSH connect error: {ex}\n{format_exc()}"
            )
            return build_unreachable_result(ctx, f"SSH error: {ex}")

        # 4) Execute checks.
        try:
            result_dict = await execute_checks(
                device_conn,
                ctx,
                read_timeout=read_timeout,
                check_timeout_overrides=check_timeout_overrides,
                check_retry_overrides=check_retry_overrides,
            )
            logger.info(f"[{device_name}] All checks done")
            return build_device_result(ctx, result_dict)
        except Exception as ex:
            logger.error(
                f"[{device_ip}] Unexpected error during checks: "
                f"{ex}\n{format_exc()}"
            )
            return build_unreachable_result(ctx, f"Unexpected error: {ex}")
        finally:
            try:
                if device_conn is not None:
                    device_conn.close()
                    await device_conn.wait_closed()
            except Exception:
                pass
