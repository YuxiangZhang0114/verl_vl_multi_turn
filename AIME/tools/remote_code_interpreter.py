import logging
import os
from typing import Any, Dict, List, Optional
from uuid import uuid4
import asyncio
import aiohttp
import json
from dataclasses import dataclass
import time

from verl.tools.base_tool import BaseTool
from verl.utils.rollout_trace import rollout_trace_op
from verl.tools.schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

from PIL import Image
import base64
from io import BytesIO

def base64_to_image(base64_str: str) -> Image.Image:
    """
    Convert a base64 string to an image.
    """
    prefix_list = [
        "data:image/png;base64,",
        "data:image/jpeg;base64,",
        "data:image/gif;base64,",
        "data:image/webp;base64,",
    ]
    for prefix in prefix_list:
        if base64_str.startswith(prefix):
            base64_str = base64_str[len(prefix) :]
            break
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    return image

def scaling_image_by_max_size(image: Image.Image, max_size: int = 512) -> Image.Image:
    """
    Scale an image to fit within a maximum size while maintaining aspect ratio.
    """
    width, height = image.size
    if width > max_size or height > max_size:
        scale = min(max_size / width, max_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        try:
            return image.resize((new_width, new_height), Image.LANCZOS)
        except AttributeError:
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

@dataclass
class RemoteServiceConfig:
    """远程服务配置"""
    service_url: str
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    # 会话管理配置
    session_timeout: int = 1800  # 30分钟
    reuse_sessions: bool = True  # 是否复用会话


class RemoteCodeInterpreter(BaseTool):
    """使用远程服务执行代码的工具
    
    该工具连接到远程的 FastAPI 服务来执行代码，提供以下功能：
    - 会话管理：支持持久化的代码执行会话
    - 错误处理：包含重试机制和详细的错误信息
    - 异步支持：使用异步 HTTP 请求提高性能
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        初始化远程代码解释器
        
        Args:
            config: 工具配置字典，应包含：
                - service_url: 远程服务的 URL
                - timeout: 请求超时时间（秒）
                - max_retries: 最大重试次数
                - retry_delay: 重试延迟时间（秒）
                - session_timeout: 会话超时时间（秒）
                - reuse_sessions: 是否复用会话
        """
        super().__init__(config, tool_schema)
        self.service_config = RemoteServiceConfig(
            service_url=config.get("service_url", "http://localhost:8001"),
            timeout=config.get("timeout", 30),
            max_retries=config.get("max_retries", 3),
            retry_delay=config.get("retry_delay", 1.0),
            session_timeout=config.get("session_timeout", 1800),
            reuse_sessions=config.get("reuse_sessions", True)
        )
        self.instance_id2session_id: Dict[str, str] = {}
        self.session_id2create_time: Dict[str, float] = {}  # 新增：记录session创建时间
        self._http_timeout = aiohttp.ClientTimeout(total=self.service_config.timeout)
        logger.info(f"初始化远程代码解释器，服务地址: {self.service_config.service_url}")

        # 启动后台清理任务
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        """创建工具实例
    
        Args:
            instance_id: 实例ID，如果未提供则自动生成
            ground_truth: 参考答案（可选）
            **kwargs: 其他参数
        
        Returns:
            str: 创建的实例ID
        """
        if not instance_id:
            instance_id = str(uuid4())
        
        try:
            session_id = await self._get_or_create_session(instance_id)
            self.instance_id2session_id[instance_id] = session_id
            logger.debug(f"成功创建实例 {instance_id}，会话 ID: {session_id}")
            return instance_id
        except Exception as e:
            logger.error(f"创建实例 {instance_id} 失败: {e}")
            raise

    async def _get_or_create_session(self, instance_id: str) -> str:
        """获取或创建会话
    
        Args:
            instance_id: 实例ID
        
        Returns:
            str: 会话ID
        """
        # 如果配置了复用会话且存在现有会话，直接返回
        if self.service_config.reuse_sessions and instance_id in self.instance_id2session_id:
            existing_session_id = self.instance_id2session_id[instance_id]
            logger.debug(f"复用现有会话 {existing_session_id} for instance {instance_id}")
            return existing_session_id
        
        # 创建新会话
        logger.debug(f"为实例 {instance_id} 创建新会话")
        return await self._create_remote_session()

    async def _create_remote_session(self) -> str:
        """在远程服务上创建会话
    
        Returns:
            str: 新创建的会话ID
        
        Raises:
            Exception: 创建会话失败时抛出异常
        """
        url = f"{self.service_config.service_url}/sessions"
        
        for attempt in range(self.service_config.max_retries):
            try:
                async with aiohttp.ClientSession(timeout=self._http_timeout) as session:
                    async with session.post(url) as response:
                        # 成功创建会话
                        if response.status == 200:
                            data = await response.json()
                            session_id = data["session_id"]
                            self.session_id2create_time[session_id] = time.time()  # 记录创建时间
                            logger.debug(f"成功创建远程会话: {session_id}")
                            return session_id
                        
                        # 服务过载，使用指数退避重试
                        elif response.status == 429:
                            if attempt < self.service_config.max_retries - 1:
                                wait_time = self.service_config.retry_delay * (2 ** attempt)
                                logger.warning(f"服务过载，等待 {wait_time} 秒后重试... (尝试 {attempt + 1}/{self.service_config.max_retries})")
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                raise Exception("服务过载，所有重试都失败")
                        
                        # 其他HTTP错误
                        else:
                            error_text = await response.text()
                            raise Exception(f"创建会话失败: HTTP {response.status} - {error_text}")
                        
            except asyncio.TimeoutError:
                logger.warning(f"创建会话超时，尝试 {attempt + 1}/{self.service_config.max_retries}")
                if attempt < self.service_config.max_retries - 1:
                    await asyncio.sleep(self.service_config.retry_delay)
                else:
                    raise Exception("创建会话超时，所有重试都失败")
                
            except aiohttp.ClientError as e:
                logger.error(f"HTTP客户端错误: {e}")
                if attempt < self.service_config.max_retries - 1:
                    await asyncio.sleep(self.service_config.retry_delay)
                else:
                    raise Exception(f"HTTP客户端错误，所有重试都失败: {e}")
                
            except Exception as e:
                logger.error(f"创建会话时发生未知错误: {e}")
                if attempt < self.service_config.max_retries - 1:
                    await asyncio.sleep(self.service_config.retry_delay)
                else:
                    raise Exception(f"创建会话失败: {e}")
    
        # 理论上不会到达这里，但作为保险
        raise Exception("创建会话失败：未知错误")

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        """执行代码"""
        # raise NotImplementedError("This method should be implemented in subclasses.")
        
        code = parameters.get("code", "")
        if not isinstance(code, str):
            code = str(code)
        
        if not code.strip():
            return "错误：代码为空", None, None
        
        # 获取会话ID
        session_id = self.instance_id2session_id.get(instance_id)
        if not session_id:
            raise Exception(f"实例 {instance_id} 没有关联的会话，请先创建会话")
        

        
        # 执行代码
        try:
            result = await self._execute_code_remote(session_id, code)
            
            return result, None, None
            
        except Exception as e:
            logger.error(f"代码执行失败: {e}")
            return f"代码执行失败: {str(e)}", None, None

    async def _execute_code_remote(self, session_id: str, code: str) -> str:
        """在远程服务上执行代码"""
        # raise NotImplementedError("This method should be implemented in subclasses.")
        
        url = f"{self.service_config.service_url}/sessions/{session_id}/execute"
        payload = {"code": code}
        
        for attempt in range(self.service_config.max_retries):
            try:
                async with aiohttp.ClientSession(timeout=self._http_timeout) as session:
                    async with session.post(url, json=payload) as response:
                        if response.status == 200:
                            data = await response.json()
                            return self._format_execution_result(data["content_to_agent"])
                        else:
                            error_text = await response.text()
                            raise Exception(f"执行代码失败: {response.status} - {error_text}")
            
            except asyncio.TimeoutError:
                logger.warning(f"代码执行超时，尝试 {attempt + 1}/{self.service_config.max_retries}")
                if attempt < self.service_config.max_retries - 1:
                    await asyncio.sleep(self.service_config.retry_delay)
                else:
                    raise Exception("代码执行超时，所有重试都失败")
            
            except Exception as e:
                logger.error(f"代码执行时发生错误: {e}")
                if attempt < self.service_config.max_retries - 1:
                    await asyncio.sleep(self.service_config.retry_delay)
                else:
                    raise

    def _format_execution_result(self, content_to_agent: List[Dict[str, Any]]) -> str:
        """格式化执行结果"""

        result_parts = []
        for item in content_to_agent:
            item_type = item.get("type", "unknown")

            if item_type == "text":
                text = item.get("text", "")
                result_parts.append({'type': 'text', 'text': text})
            elif item_type == "image":
                image_url = item.get("image_url")
                image = base64_to_image(image_url)
                image = scaling_image_by_max_size(image ,max_size=512)
                
                result_parts.append({'type': 'image', 'image': image})
        return result_parts
                




    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        """计算奖励"""
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        """释放工具实例"""
        if instance_id not in self.instance_id2session_id:
            logger.warning(f"实例 {instance_id} 不存在或已被释放，无法释放")
            return 
        session_id = self.instance_id2session_id[instance_id]
        try:
            await self._delete_remote_session(session_id)
            self.instance_id2session_id.pop(instance_id, None)
            self.session_id2create_time.pop(session_id, None)  # 删除创建时间记录
            logger.debug(f"成功释放实例 {instance_id}，删除远程会话 {session_id}")
        except Exception as e:
            logger.error(f"释放实例 {instance_id} 时删除远程会话失败: {e}")

    async def _delete_remote_session(self, session_id: str) -> None:
        """删除远程会话"""
        url = f"{self.service_config.service_url}/sessions/{session_id}"
        
        try:
            async with aiohttp.ClientSession(timeout=self._http_timeout) as session:
                async with session.delete(url) as response:
                    if response.status == 200:
                        logger.debug(f"成功删除远程会话 {session_id}")
                    else:
                        logger.warning(f"删除远程会话失败: {response.status}")
        except Exception as e:
            logger.error(f"删除远程会话时发生错误: {e}")

    async def _cleanup_expired_sessions(self):
        """定期清理超时的 session"""
        while True:
            now = time.time()
            expired_sessions = [
                sid for sid, ctime in self.session_id2create_time.items()
                if now - ctime > self.service_config.session_timeout
            ]
            for sid in expired_sessions:
                try:
                    await self._delete_remote_session(sid)
                    self.session_id2create_time.pop(sid, None)
                    # 同时清理 instance_id2session_id
                    for iid, ssid in list(self.instance_id2session_id.items()):
                        if ssid == sid:
                            self.instance_id2session_id.pop(iid, None)
                    logger.info(f"自动清理超时 session: {sid}")
                except Exception as e:
                    logger.error(f"自动清理 session {sid} 失败: {e}")
            await asyncio.sleep(60)  # 每分钟检查一次

    async def _delete_all_sessions(self):
        """删除所有已创建的远程会话"""
        for session_id in list(self.session_id2create_time.keys()):
            try:
                await self._delete_remote_session(session_id)
                logger.info(f"销毁时自动删除 session: {session_id}")
            except Exception as e:
                logger.error(f"销毁时删除 session {session_id} 失败: {e}")
        self.session_id2create_time.clear()
        self.instance_id2session_id.clear()

    def __del__(self):
        """对象销毁时自动清理所有 session"""
        try:
            # 由于 __del__ 不能直接调用异步方法，这里用 asyncio.run
            if self.session_id2create_time:
                asyncio.run(self._delete_all_sessions())
        except Exception as e:
            logger.error(f"对象销毁时自动清理 session 失败: {e}")

