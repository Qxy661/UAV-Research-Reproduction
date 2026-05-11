"""
LLM 任务规划本地评估（无需 API）
测试任务分解逻辑的正确性框架
"""

import json
import os
import sys
import time
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_task_to_steps(task_text):
    """规则引擎：从任务文本提取飞行步骤（模拟 LLM 输出）"""
    steps = []
    task_lower = task_text.lower()

    # 起飞
    if "起飞" in task_text:
        alt_match = re.search(r'(\d+)\s*米', task_text)
        alt = int(alt_match.group(1)) if alt_match else 10
        steps.append({"action": "takeoff", "params": {"altitude": alt}})

    # goto 坐标
    goto_pattern = r'[A-Z]?\(?(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)?'
    for match in re.finditer(goto_pattern, task_text):
        x, y, z = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if not steps or steps[-1].get("params", {}).get("x") != x:
            steps.append({"action": "goto", "params": {"x": x, "y": y, "z": z}})

    # 悬停
    if "悬停" in task_text:
        dur_match = re.search(r'悬停\s*(\d+)\s*秒', task_text)
        dur = int(dur_match.group(1)) if dur_match else 30
        steps.append({"action": "hover", "params": {"duration": dur}})

    # 绕圈
    if "绕" in task_text and ("圈" in task_text or "飞行" in task_text):
        lap_match = re.search(r'(\d+)\s*圈', task_text)
        laps = int(lap_match.group(1)) if lap_match else 3
        steps.append({"action": "loiter", "params": {"laps": laps}})

    # 网格搜索
    if "网格" in task_text or "之字形" in task_text or "扫描" in task_text:
        steps.append({"action": "grid_search", "params": {"pattern": "zigzag"}})

    # 拍照
    if "拍照" in task_text or "拍摄" in task_text:
        steps.append({"action": "capture_image", "params": {}})

    # 编队
    if "编队" in task_text:
        count_match = re.search(r'(\d+)\s*架', task_text)
        count = int(count_match.group(1)) if count_match else 3
        steps.append({"action": "formation_takeoff", "params": {"count": count}})

    # 跟踪
    if "跟踪" in task_text or "跟随" in task_text:
        steps.append({"action": "follow_target", "params": {}})

    # 覆盖
    if "覆盖" in task_text:
        steps.append({"action": "area_coverage", "params": {}})

    # 巡逻
    if "巡逻" in task_text or "往返" in task_text:
        steps.append({"action": "patrol", "params": {}})

    # 降落
    if "降落" in task_text or "返回" in task_text:
        steps.append({"action": "land", "params": {}})

    return steps


def evaluate_anomaly_response(scenario, expected_response):
    """规则引擎：评估异常处理响应"""
    response_lower = expected_response.lower()
    scenario_lower = scenario.lower()

    checks = {
        "gps": ("gps" in scenario_lower, any(kw in response_lower for kw in ["定位", "imu", "视觉", "悬停"])),
        "电池": ("电量" in scenario_lower or "电池" in scenario_lower, any(kw in response_lower for kw in ["返航", "降落", "紧急"])),
        "障碍物": ("障碍" in scenario_lower, any(kw in response_lower for kw in ["避障", "绕行", "悬停", "减速"])),
        "通信": ("通信" in scenario_lower or "丢包" in scenario_lower, any(kw in response_lower for kw in ["降低", "返航", "自主"])),
        "风速": ("风" in scenario_lower, any(kw in response_lower for kw in ["降低", "降落", "高度"])),
        "传感器": ("imu" in scenario_lower or "传感器" in scenario_lower, any(kw in response_lower for kw in ["冗余", "降落", "切换"])),
        "温度": ("温度" in scenario_lower or "过热" in scenario_lower, any(kw in response_lower for kw in ["降低", "降落", "冷却"])),
        "禁飞": ("禁飞" in scenario_lower, any(kw in response_lower for kw in ["停止", "绕行", "重新规划"])),
        "超时": ("超时" in scenario_lower or "时间" in scenario_lower, any(kw in response_lower for kw in ["电量", "返航", "评估"])),
        "云台": ("云台" in scenario_lower or "相机" in scenario_lower, any(kw in response_lower for kw in ["姿态", "调整", "跳过"])),
    }

    for name, (triggered, handled) in checks.items():
        if triggered:
            return handled
    return False


def main():
    """主评估函数"""
    test_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_cases")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    # 加载测试用例
    with open(os.path.join(test_dir, "missions.json"), "r", encoding="utf-8") as f:
        missions = json.load(f)
    with open(os.path.join(test_dir, "anomalies.json"), "r", encoding="utf-8") as f:
        anomalies = json.load(f)

    print("=" * 60)
    print("LLM 任务规划本地评估（规则引擎模拟）")
    print("=" * 60)

    # 任务分解评估
    print("\n[Task Decomposition Evaluation]")
    task_results = []
    correct = 0

    for case in missions:
        start = time.time()
        parsed = parse_task_to_steps(case["task"])
        latency = time.time() - start

        parsed_actions = set(s["action"] for s in parsed)
        expected_actions = set(s.split("(")[0] for s in case["expected_steps"])
        matched = parsed_actions & expected_actions
        match_rate = len(matched) / len(expected_actions) if expected_actions else 0
        is_correct = match_rate >= 0.6

        if is_correct:
            correct += 1

        task_results.append({
            "task_id": case["id"],
            "difficulty": case["difficulty"],
            "expected": list(expected_actions),
            "parsed": list(parsed_actions),
            "match_rate": match_rate,
            "correct": is_correct,
            "latency": latency,
        })

        status = "PASS" if is_correct else "FAIL"
        print(f"  {status} 任务{case['id']:2d} [{case['difficulty']:6s}] 匹配率={match_rate:.0%} | {case['task'][:40]}...")

    accuracy = correct / len(missions)
    print(f"\n  总准确率: {correct}/{len(missions)} = {accuracy:.1%}")

    # 按难度统计
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in task_results if r["difficulty"] == diff]
        if subset:
            diff_acc = sum(r["correct"] for r in subset) / len(subset)
            print(f"  {diff}: {sum(r['correct'] for r in subset)}/{len(subset)} = {diff_acc:.1%}")

    # 异常处理评估
    print("\n[Anomaly Handling Evaluation]")
    anomaly_results = []
    anomaly_correct = 0

    for case in anomalies:
        handled = evaluate_anomaly_response(case["scenario"], case["expected_response"])
        if handled:
            anomaly_correct += 1

        anomaly_results.append({
            "anomaly_id": case["id"],
            "scenario": case["scenario"],
            "severity": case["severity"],
            "handled": handled,
        })

        status = "PASS" if handled else "FAIL"
        print(f"  {status} 异常{case['id']:2d} [{case['severity']:8s}] {case['scenario'][:45]}")

    anomaly_rate = anomaly_correct / len(anomalies)
    print(f"\n  总检测率: {anomaly_correct}/{len(anomalies)} = {anomaly_rate:.1%}")

    # 汇总报告
    report = {
        "task_decomposition": {
            "total": len(missions),
            "correct": correct,
            "accuracy": accuracy,
            "by_difficulty": {},
        },
        "anomaly_handling": {
            "total": len(anomalies),
            "correct": anomaly_correct,
            "detection_rate": anomaly_rate,
        },
    }

    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in task_results if r["difficulty"] == diff]
        if subset:
            report["task_decomposition"]["by_difficulty"][diff] = {
                "total": len(subset),
                "correct": sum(r["correct"] for r in subset),
                "accuracy": sum(r["correct"] for r in subset) / len(subset),
            }

    with open(os.path.join(output_dir, "evaluation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "task_results.json"), "w", encoding="utf-8") as f:
        json.dump(task_results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "anomaly_results.json"), "w", encoding="utf-8") as f:
        json.dump(anomaly_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"评估完成，结果保存至: {output_dir}")
    print(f"{'='*60}")

    return report


if __name__ == "__main__":
    main()
