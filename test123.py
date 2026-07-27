# cat unicorn_server.py
# Reconstructed from your screenshots -- verify against your original.
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from typing import Annotated, Literal
import requests
from requests.adapters import HTTPAdapter
from environs import Env
import yaml

env = Env()
env.read_env()

mcp = FastMCP("unicorn-server")


class Settings:
    # _basedir = os.path.abspath(os.path.dirname(__file__))

    TRAP_HTTP_EXCEPTIONS = True
    ERROR_404_HELP = True
    BUNDLE_ERRORS = True

    try:
        with open("credentials.yml", "r") as f:
            credentials = yaml.safe_load(f)
    except FileNotFoundError:
        credentials = {}

    UNICORN_URLS = credentials.get("UNICORN_URLS", {})
    UNICORN_TOKEN = credentials.get("UNICORN_TOKEN", {})
    PROXIES = credentials.get("PROXIES", {})

    SOGPT_CHAT_COMPLETION_URL = env("SOGPT_CHAT_COMPLETION_URL")

    CLIENT_ID = env("CLIENT_ID")
    CLIENT_SECRET = env("CLIENT_SECRET")
    X_APPLICATION = env("X-APPLICATION")
    X_KEY_NAME = env("X-KEY-NAME")

    UNITY_CLIENT_ID = env("UNITY_CLIENT_ID")
    UNITY_CLIENT_SECRET = env("UNITY_CLIENT_SECRET")
    UNITY_SCOPE = env("UNITY_SCOPE")
    UNITY_ENV = env("UNITY_ENV")


@mcp.tool(
    name="get_device_details",
    description="""
    This tool used to search the device details from unicorn (CMDB).
    It give the device details based on device name and region provided by the user
    """)
def get_unicorn_device_details(
    device_name: Annotated[str, Field(description="name of the device")],
    region: Annotated[str | None, Field(
        description="Region (PARIS, ASIA, AMER, UK, INDIA, IBFS). "
                    "Omit or set 'AUTO' to auto-detect")] = None,
) -> dict | str:
    try:
        with requests.Session() as session:
            session.mount('https://', HTTPAdapter(max_retries=3))

            def fetch(region_name: str) -> dict | None:
                for proxy in Settings.PROXIES:
                    try:
                        begin, end = proxy.get("uri").split("//")
                        proxy_info = {
                            proxy.get("protocol"):
                                f"{begin}//{proxy.get('login')}:{proxy.get('password')}@{end}"
                        }
                        url = f"{Settings.UNICORN_URLS[region_name]}/{device_name.lower()}"
                        headers = {
                            'Authorization': f"Token {Settings.UNICORN_TOKEN[region_name]}",
                            'Accept': 'application/json'
                        }
                        response = session.get(
                            url,
                            verify=False,
                            headers=headers,
                            proxies=proxy_info if region_name not in ['ASIA', 'AMER', 'INDIA'] else None,
                            timeout=(3, 60),
                        )
                        if response.status_code in (200, 201):
                            return response.json()
                    except Exception as ex:
                        # Continue with next proxy or region
                        print(f"[{region_name}] proxy attempt failed: {ex}")
                return None

            requested_region = (region or "").strip().upper()
            if not requested_region or requested_region == "AUTO":
                for candidate in Settings.UNICORN_URLS.keys():
                    data = fetch(candidate)
                    if data:
                        return {"region": candidate, "data": data}
                return "No data found in any region."
            else:
                if requested_region not in Settings.UNICORN_URLS:
                    return (f"Invalid region '{requested_region}'. "
                            f"Valid: {', '.join(Settings.UNICORN_URLS.keys())}")
                data = fetch(requested_region)
                return {"region": requested_region, "data": data} if data else "No data found!!"
    except Exception as ex:
        return f"Error: {str(ex)}"


if __name__ == "__main__":
    # mcp.run(
    #     transport="sse",
    #     host="175.60.57.250",
    #     port=4200,
    #     path="/unicorn_server",
    #     log_level="debug"
    # )
    mcp.run()
