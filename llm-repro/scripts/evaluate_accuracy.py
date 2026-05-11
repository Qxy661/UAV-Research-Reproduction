"""
LLM 任务规划准确率评估脚本
测试不同 LLM 在无人机任务分解和异常处理上的表现
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.task_planner import TaskPlanner, evaluate_task_decomposition


def run_task_evaluation(planner, test_cases):
    """评估任务分解准确率"""
    results = []

    for case in test_cases:
        print(f"  测试任务 {case['id']}: {case['task'][:50]}...")

        result = planner.decompose_task(
            case["task"],
            safety_constraints=case.get("safety_constraints"),
        )

        evaluation = evaluate_task_decomposition(
            case["expected_steps"], result
        )

        results.append({
            "task_id": case["id"],
            "task": case["task"],
            "difficulty": case["difficulty"],
            "expected_steps": case["expected_steps"],
            "actual_result": result,
            "evaluation": evaluation,
        })

        # 避免 API 限速
        time.sleep(0.5)

    return results


def run_anomaly_evaluation(planner, anomaly_cases):
    """评估异常处理能力"""
    results = []

    for case in anomaly_cases:
        print(f"  测试异常 {case['id']}: {case['scenario'][:50]}...")

        result = planner.handle_anomaly(
            case["scenario"],
            case["context"],
        )

        # 简单评估：是否包含关键词
        response_text = json.dumps(result, ensure_ascii=False).lower()
        detected_keywords = [
            kw for kw in case["detection_keywords"]
            if kw.lower() in response_text
        ]

        results.append({
            "anomaly_id": case["id"],
            "scenario": case["scenario"],
            "severity": case["severity"],
            "result": result,
            "keywords_detected": detected_keywords,
            "keyword_detection_rate": len(detected_keywords) / len(case["detection_keywords"]),
        })

        time.sleep(0.5)

    return results


def generate_report(task_results, anomaly_results, latency_stats, output_dir):
    """生成评估报告"""

    # 任务分解统计
    total_tasks = len(task_results)
    correct_tasks = sum(1 for r in task_results if r["evaluation"]["correct"])
    accuracy = correct_tasks / total_tasks if total_tasks > 0 else 0

    # 按难度统计
    difficulty_stats = {}
    for r in task_results:
        diff = r["difficulty"]
        if diff not in difficulty_stats:
            difficulty_stats[diff] = {"total": 0, "correct": 0}
        difficulty_stats[diff]["total"] += 1
        if r["evaluation"]["correct"]:
            difficulty_stats[diff]["correct"] += 1

    # 异常处理统计
    total_anomalies = len(anomaly_results)
    avg_keyword_rate = (
        sum(r["keyword_detection_rate"] for r in anomaly_results) / total_anomalies
        if total_anomalies > 0
        else 0
    )

    # 生成报告
    report = {
        "task_decomposition": {
            "total": total_tasks,
            "correct": correct_tasks,
            "accuracy": accuracy,
            "by_difficulty": {
                diff: {
                    "total": stats["total"],
                    "correct": stats["correct"],
                    "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
                }
                for diff, stats in difficulty_stats.items()
            },
        },
        "anomaly_handling": {
            "total": total_anomalies,
            "avg_keyword_detection_rate": avg_keyword_rate,
        },
        "latency": latency_stats,
    }

    # 保存报告
    with open(os.path.join(output_dir, "evaluation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 打印报告
    print("\n" + "=" * 60)
    print("LLM 任务规划评估报告")
    print("=" * 60)

    print(f"\n📊 任务分解准确率:")
    print(f"  总任务数: {total_tasks}")
    print(f"  正确数: {correct_tasks}")
    print(f"  准确率: {accuracy:.1%}")

    print(f"\n📊 按难度统计:")
    for diff, stats in difficulty_stats.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {diff}: {stats['correct']}/{stats['total']} ({acc:.1%})")

    print(f"\n📊 异常处理:")
    print(f"  总异常数: {total_anomalies}")
    print(f"  平均关键词检测率: {avg_keyword_rate:.1%}")

    print(f"\n📊 延迟统计:")
    if latency_stats:
        print(f"  平均延迟: {latency_stats['mean']:.2f}s")
        print(f"  P90 延迟: {latency_stats['p90']:.2f}s")
        print(f"  P95 延迟: {latency_stats['p95']:.2f}s")
        print(f"  最小延迟: {latency_stats['min']:.2f}s")
        print(f"  最大延迟: {latency_stats['max']:.2f}s")

    print("\n" + "=" * 60)

    return report


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="LLM 任务规划评估")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic", "local"])
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--output-dir", default="results/evaluation")
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载测试用例
    test_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_cases")
    with open(os.path.join(test_dir, "missions.json"), "r", encoding="utf-8") as f:
        missions = json.load(f)
    with open(os.path.join(test_dir, "anomalies.json"), "r", encoding="utf-8") as f:
        anomalies = json.load(f)

    # 创建规划器
    planner = TaskPlanner(provider=args.provider, model=args.model)

    print(f"开始评估: {args.provider}/{args.model}")
    print(f"测试任务: {len(missions)} 个")
    print(f"异常场景: {len(anomalies)} 个")
    print()

    # 运行评估
    print("📝 评估任务分解...")
    task_results = run_task_evaluation(planner, missions)

    print("\n🚨 评估异常处理...")
    anomaly_results = run_anomaly_evaluation(planner, anomalies)

    # 生成报告
    latency_stats = planner.get_latency_stats()
    report = generate_report(task_results, anomaly_results, latency_stats, args.output_dir)

    # 保存详细结果
    with open(os.path.join(args.output_dir, "task_results.json"), "w", encoding="utf-8") as f:
        json.dump(task_results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "anomaly_results.json"), "w", encoding="utf-8") as f:
        json.dump(anomaly_results, f, indent=2, ensure_ascii=False)

    print(f"\n详细结果已保存至: {args.output_dir}")


if __name__ == "__main__":
    main()
