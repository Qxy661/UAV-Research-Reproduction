"""
LLM 任务规划器 — 将自然语言任务分解为可执行步骤
支持 OpenAI / Anthropic / 本地模型 API
"""

import json
import time
import os


# 无人机 API 定义（用于 prompt）
UAV_API_DOCS = """
## 可用的无人机 API

```python
# 基础飞行
takeoff(altitude=10)                          # 起飞到指定高度
land()                                        # 降落
goto(x, y, z, speed=5)                        # 飞到指定坐标
hover(duration=30)                            # 悬停指定秒数

# 航线任务
loiter(center=(x,y,z), radius=10, direction='cw', laps=3)  # 绕圈飞行
waypoints(points=[(x1,y1,z1), ...], speed=5)  # 按航点飞行
grid_search(area=WxH, spacing=S, altitude=H, pattern='zigzag')  # 网格搜索
area_coverage(area=WxH, overlap=0.3, altitude=H)  # 区域覆盖

# 高级功能
capture_image()                               # 拍照
start_recording()                             # 开始录像
follow_target(target, altitude, camera)       # 跟踪目标
formation_takeoff(count, pattern)             # 编队起飞
formation_goto(target)                        # 编队飞行

# 安全检查
check_battery(threshold=20)                   # 检查电量
return_to_launch()                            # 返航
emergency_land()                              # 紧急降落
```

## 输出格式

请将任务分解为可执行步骤，输出 JSON 格式：
```json
{
  "steps": [
    {"action": "takeoff", "params": {"altitude": 10}},
    {"action": "goto", "params": {"x": 10, "y": 20, "z": 5}},
    ...
  ],
  "safety_checks": ["max_altitude:50", "max_speed:5"],
  "estimated_time": 120,
  "estimated_distance": 200
}
```
"""

# 异常处理 prompt
ANOMALY_PROMPT = """
你是一个无人机异常处理专家。当检测到以下异常时，请给出处理建议：

## 异常信息
{anomaly_info}

## 当前上下文
{context}

## 可用的安全操作
- hover(): 立即悬停
- emergency_land(): 紧急降落
- return_to_launch(): 返航
- reduce_altitude(height): 降低高度
- reduce_speed(speed): 降低速度
- replan_route(): 重新规划路径

请输出 JSON 格式的处理建议：
```json
{
  "severity": "critical|high|medium|low",
  "action": "建议的操作",
  "reasoning": "决策推理过程",
  "steps": ["step1", "step2", ...]
}
```
"""


class TaskPlanner:
    """LLM 任务规划器"""

    def __init__(self, provider="openai", model="gpt-4o", api_key=None):
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.environ.get(f"{provider.upper()}_API_KEY", "")

        # 延迟统计
        self.latencies = []

    def decompose_task(self, task_description, safety_constraints=None):
        """将自然语言任务分解为可执行步骤"""

        prompt = f"""你是一个无人机任务规划专家。请将以下自然语言任务分解为可执行的飞行步骤。

## 任务描述
{task_description}

## 安全约束
{json.dumps(safety_constraints or [], ensure_ascii=False)}

{UAV_API_DOCS}

请只输出 JSON，不要有其他文字。"""

        # 调用 API
        start_time = time.time()
        response = self._call_api(prompt)
        latency = time.time() - start_time
        self.latencies.append(latency)

        # 解析响应
        try:
            result = json.loads(response)
            result["latency"] = latency
            return result
        except json.JSONDecodeError:
            return {
                "error": "JSON 解析失败",
                "raw_response": response,
                "latency": latency,
            }

    def handle_anomaly(self, anomaly_info, context):
        """处理异常情况"""

        prompt = ANOMALY_PROMPT.format(
            anomaly_info=anomaly_info,
            context=context,
        )

        start_time = time.time()
        response = self._call_api(prompt)
        latency = time.time() - start_time
        self.latencies.append(latency)

        try:
            result = json.loads(response)
            result["latency"] = latency
            return result
        except json.JSONDecodeError:
            return {
                "error": "JSON 解析失败",
                "raw_response": response,
                "latency": latency,
            }

    def _call_api(self, prompt):
        """调用 LLM API"""
        if self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt)
        elif self.provider == "local":
            return self._call_local(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_openai(self, prompt):
        """调用 OpenAI API"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是无人机任务规划专家，只输出JSON格式。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1000,
            )

            return response.choices[0].message.content
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _call_anthropic(self, prompt):
        """调用 Anthropic API"""
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=self.api_key)

            response = client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )

            return response.content[0].text
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _call_local(self, prompt):
        """调用本地模型 API（如 Ollama）"""
        try:
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=30,
            )
            return response.json()["response"]
        except Exception as e:
            return json.dumps({"error": str(e)})

    def get_latency_stats(self):
        """获取延迟统计"""
        if not self.latencies:
            return {}

        import numpy as np
        latencies = np.array(self.latencies)
        return {
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p90": float(np.percentile(latencies, 90)),
            "p95": float(np.percentile(latencies, 95)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies)),
            "count": len(self.latencies),
        }


def evaluate_task_decomposition(expected, actual):
    """评估任务分解准确率"""

    if "error" in actual:
        return {"correct": False, "error": actual["error"]}

    # 提取实际步骤
    actual_steps = actual.get("steps", [])

    # 简单匹配（检查关键动作是否出现）
    expected_actions = set()
    for step in expected:
        action = step.split("(")[0]  # 提取动作名
        expected_actions.add(action)

    actual_actions = set()
    for step in actual_steps:
        if isinstance(step, dict):
            actual_actions.add(step.get("action", ""))
        elif isinstance(step, str):
            actual_actions.add(step.split("(")[0])

    # 计算匹配率
    if not expected_actions:
        return {"correct": True, "match_rate": 1.0}

    matched = expected_actions & actual_actions
    match_rate = len(matched) / len(expected_actions)

    return {
        "correct": match_rate >= 0.8,  # 80% 匹配率算正确
        "match_rate": match_rate,
        "expected_actions": list(expected_actions),
        "actual_actions": list(actual_actions),
        "matched_actions": list(matched),
    }
