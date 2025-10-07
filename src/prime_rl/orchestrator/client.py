import asyncio
import os
from pathlib import Path

import httpx
import json
from httpx import Response
from openai import AsyncOpenAI, NotFoundError

from prime_rl.orchestrator.config import ClientConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import get_weight_ckpt_model_path


def setup_client(client_config: ClientConfig) -> AsyncOpenAI:
    # We use a longer request timeout than default, but if more than 20min, we probably need faster inference deployment
    timeout = httpx.Timeout(timeout=client_config.timeout, connect=5.0)
    # We use as many concurrent connections as possible, but lower than available ports
    limits = httpx.Limits(
        max_connections=28000,  # OAI default: 1000
        max_keepalive_connections=28000,  # OAI default: 100
    )
    http_client = httpx.AsyncClient(limits=limits, timeout=timeout)
    return AsyncOpenAI(
        base_url=client_config.base_url,
        api_key=os.getenv(client_config.api_key_var, "EMPTY"),
        max_retries=10,  # OAI default: 2 (does exponential backoff and reasonable timeout in between retries)
        http_client=http_client,
    )


async def check_health(client: AsyncOpenAI, interval: int = 1, log_interval: int = 10, timeout: int = 1800) -> None:
    logger = get_logger()
    wait_time = 0
    url = str(client.base_url).strip()[:-4] + "/health"
    logger.debug(f"Starting pinging {url} to check health")
    while wait_time < timeout:
        try:
            await client.get(url, cast_to=Response, options={"max_retries": 0})
            logger.debug(f"Inference pool is ready after {wait_time} seconds")
            return
        except NotFoundError:
            logger.warning(f"The route {url} does not exist. Skipping health check.")
            return
        except Exception as e:
            if wait_time % log_interval == 0 and wait_time > 0:
                logger.warning(f"Inference server was not reached after {wait_time} seconds (Error: {e})")
            await asyncio.sleep(interval)
            wait_time += interval
    msg = f"Inference server is not ready after {wait_time} (>{timeout}) seconds. Aborting..."
    logger.error(msg)
    raise TimeoutError(msg)


async def check_has_model(client: AsyncOpenAI, model_name: str) -> None:
    logger = get_logger()
    logger.debug(f"Checking if model {model_name} is in the inference pool")
    models = (await client.models.list()).data
    if not any(model.id == model_name for model in models):
        raise ValueError(f"Model {model_name} was not found in the inference pool")
    logger.debug(f"Model {model_name} was found in the inference pool")


# async def update_weights(client: AsyncOpenAI, path: Path, step: int) -> None:
#     """Make a HTTP post request to the vLLM server to update the weights."""
#     logger = get_logger()
#     url = str(client.base_url).strip()[:-4] + "/update_weights"
#     try:
#         model_path = get_weight_ckpt_model_path(path, step).absolute()
#         logger.debug(f"Sending request to {url} to update weights from {model_path}")
#         # print(f"Sending request to {url} to update weights from {model_path}")
#         # Sending request to http://127.0.0.1:30000/update_weights to update weights from /scratch/luogeng/project/prime-rl/outputs/weights/step_1/pytorch_model.bin
#         await client.post(url, cast_to=Response, body={"model_path": model_path.as_posix()})
#     except NotFoundError:
#         logger.warning(f"The route {url} does not exist. Skipping weight update.")
#         return

async def update_weights(client: AsyncOpenAI, path: Path, step: int) -> None:
    """
    Upload a single .pt weights file to the vLLM server via multipart/form-data.
    Matches the new inference /update_weights endpoint (MVP).
    """
    logger = get_logger()
    url = str(client.base_url).strip()[:-4] + "/update_weights"

    # 取得本地权重文件路径（例：.../outputs/weights/step_{step}/pytorch_model.bin）
    model_path = get_weight_ckpt_model_path(path, step).absolute()
    logger.debug(f"Uploading weights for step {step} from {model_path} to {url}")
    print(f"Uploading weights for step {step} from {model_path} to {url}")

    # 说明：
    # - 我们用独立的 httpx.AsyncClient 发 multipart（OpenAI 客户端默认是 JSON）
    # - 连接超时保持 5s，读超时给足（1 小时）以覆盖 1GB 级上传
    timeout = httpx.Timeout(connect=5.0, read=3600.0, write=3600.0, pool=3600.0)

    limits = httpx.Limits(max_connections=10, max_keepalive_connections=10)

    try:
        # 以流式方式上传文件：不要把 1GB 读进内存
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as ac:
            # metadata 放 JSON 字符串；format 固定 "pt"（MVP）
            metadata = {"step": step, "format": "pt"}

            # 用文件句柄做流式上传；务必按二进制方式打开
            with open(model_path, "rb") as fh:
                data = {"metadata": json.dumps({"step": step, "format": "pt"})}
                files = {"file": (model_path.name, fh, "application/octet-stream")}
                resp = await ac.post(url, data=data, files=files)

            # 统一错误处理
            if resp.status_code >= 400:
                try:
                    detail = resp.json()
                except Exception:
                    detail = resp.text
                raise RuntimeError(f"update_weights failed ({resp.status_code}): {detail}")

            # 日志记录成功
            logger.debug(f"update_weights OK: {resp.text}")

    except NotFoundError:
        # 兼容 inference 未实现该路由的情况（比如老版本）
        logger.warning(f"The route {url} does not exist. Skipping weight update.")
        return


async def reload_weights(client: AsyncOpenAI) -> None:
    """Make a HTTP post request to the vLLM server to reload weights (reset to base model)."""
    logger = get_logger()
    url = str(client.base_url).strip()[:-4] + "/reload_weights"
    try:
        logger.debug(f"Sending request to {url} to reload weights (reset to base model)")
        await client.post(url, cast_to=Response, body={})
    except NotFoundError:
        logger.warning(f"The route {url} does not exist. Skipping weight reload.")
        return
    await client.post(url, cast_to=Response, body={})
