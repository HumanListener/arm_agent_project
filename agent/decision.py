"""
决策层：把 YOLO 的结构化检测结果变成"可执行的处置方案 + 人类可读的决策报告"。

两级决策，保证演示永远不卡：
  1) 规则脑（rule_decide）：确定性映射，永远可用；
  2) 大模型脑（llm_decide）：调用本地 Ollama，输出更"像质检工程师"的理由，
     并用 LLM 的结果覆盖规则脑的 reason / severity（dest 仍走规则表，避免模型乱选坑位）。
Ollama 不在线 / 超时 / 输出不是 JSON → 自动降级到规则脑。

返回的 decision 统一结构：
{
  "action":     "SORT" | "REVIEW" | "PASS",       # 该干什么
  "dest":       "ng" | "rework" | "review" | "",  # 投放到哪个区（PASS 时为空）
  "dest_label": "废品区" | ...,                    # 给人和界面看的中文名
  "reason":     "一句话说明为什么这么判",
  "severity":   "high" | "medium" | "low" | "none",
  "report":     "面向操作工的完整中文决策报告（可含建议动作）",
  "engine":     "rule" | "ollama:<model>",        # 这条决策是谁给出的，便于溯源
}
"""
from __future__ import annotations

import json
import re
import time
from typing import Optional

import requests

from config.settings import (
    OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_FALLBACK_MODELS,
    YOLO_CONF_THRES, DEFECT_HIGH_CONF, DEST_BINS,
)

# ---------------- 规则知识库（NEU-DET 六类缺陷）----------------
# severity: high=直接报废 / medium=返修 / low=可接受但需复检
DEFECT_RULES: dict[str, dict] = {
    "crazing":        {"cn": "龟裂",     "severity": "medium", "dest": "rework",
                       "advice": "网状裂纹通常可通过打磨去除，返修后复检"},
    "inclusion":      {"cn": "夹杂",     "severity": "high",   "dest": "ng",
                       "advice": "非金属夹杂属冶金缺陷，无法返修，直接判废"},
    "patches":        {"cn": "斑块",     "severity": "medium", "dest": "rework",
                       "advice": "表面斑块多为氧化/油污压入，建议表面处理后复检"},
    "pitted_surface": {"cn": "麻点",     "severity": "medium", "dest": "rework",
                       "advice": "麻点为轧制氧化铁皮压入，浅层可打磨，深层判废"},
    "rolled-in_scale": {"cn": "压入氧化铁皮", "severity": "high", "dest": "ng",
                       "advice": "氧化铁皮压入较深，影响后续涂镀，判废"},
    "scratches":      {"cn": "划痕",     "severity": "high",   "dest": "ng",
                       "advice": "连续划痕破坏表面完整性，深度超差时判废"},
}

_SEVERITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


def defect_cn(cls: str) -> str:
    """英文类别名 → 中文名（未知类别原样返回）。"""
    return DEFECT_RULES.get(cls, {}).get("cn", cls or "未知缺陷")


def rule_decide(detect: dict) -> dict:
    """规则脑：纯确定性映射，任何环境下都能给出结论。"""
    if not detect.get("has_defect"):
        return {
            "action": "PASS", "dest": "", "dest_label": "",
            "reason": "本帧未检出缺陷，工件可直接流入下一工序",
            "severity": "none", "engine": "rule",
            "report": ("检测结论：无缺陷，置信度 "
                       f"{float(detect.get('conf') or 0.0) * 100:.1f}%。\n"
                       "处置：工件可直接放行，无需机械臂动作。"),
        }

    cls = detect.get("defect_cls") or ""
    conf = float(detect.get("conf") or 0.0)
    rule = DEFECT_RULES.get(cls, {"cn": cls or "未知缺陷", "severity": "medium",
                                  "dest": "review", "advice": "未知类别，转人工复检"})

    # 置信度不足 → 不做自动报废，转复检
    if conf < YOLO_CONF_THRES:
        dest = "review"
        severity = "low"
        reason = f"检出疑似{rule['cn']}但置信度仅 {conf:.2f}，不足以下判，转人工复检"
    else:
        dest = rule["dest"]
        severity = rule["severity"]
        if conf < DEFECT_HIGH_CONF and severity == "high":
            # 高严重度但置信度处于中间地带 → 降级到复检，避免误判废品
            dest, severity = "review", "medium"
            reason = (f"检出{rule['cn']}，置信度 {conf:.2f} 处于中间区间，"
                      f"先转人工复检再决定报废或返修")
        else:
            reason = f"检出{rule['cn']}（{cls}），置信度 {conf:.2f}，{rule['advice']}"

    bin_info = DEST_BINS.get(dest, DEST_BINS["review"])
    action = "SORT" if dest != "review" else "REVIEW"
    out = {
        "action": action,
        "dest": dest,
        "dest_label": bin_info["label"],
        "reason": reason,
        "severity": severity,
        "engine": "rule",
    }
    out["report"] = _compose_report(detect, out, rule_advice=rule["advice"])
    return out


# ---------------- 大模型脑 ----------------
# 注意：本地 8B 模型在 CPU 上生成一段中文报告要 1~2 分钟，所以这里只让模型输出
# "结构化结论字段"（几十个 token），完整中文报告由 Python 模板拼装，速度与可读性兼得。
_SYS_PROMPT = """你是工业钢板表面质检专家。把视觉检测结果翻译成处置决策。
只输出一个 JSON 对象，不要其他文字、不要 markdown、不要思考过程：
{"action":"SORT|REVIEW|PASS","dest":"ng|rework|review|","reason":"判据，40字以内中文","severity":"high|medium|low|none"}
判定原则：
- PASS：无缺陷，直接放行。
- SORT + dest=ng：不可修复的严重缺陷（夹杂、压入氧化铁皮、深度划痕），报废。
- SORT + dest=rework：可打磨返修的缺陷（龟裂、斑块、浅麻点），送返修。
- REVIEW + dest=review：置信度不足或严重度拿不准时转人工复检，不要冒险判废。
"""


def _compose_report(detect: dict, dec: dict, rule_advice: str = "") -> str:
    """用结论字段拼出面向现场操作工的中文决策报告（模板化，避免让 LLM 写长文）。"""
    cls_en = detect.get("defect_cls") or ""
    cn = defect_cn(cls_en) if cls_en else "无"
    conf = float(detect.get("conf") or 0.0)
    sev_cn = {"high": "高", "medium": "中", "low": "低", "none": "无"}.get(dec.get("severity", ""), "未知")
    action = dec.get("action", "PASS")
    dest_label = dec.get("dest_label") or ""
    head = (f"检测结论：{cn}"
            + (f"（{cls_en}）" if cls_en else "")
            + f"，置信度 {conf * 100:.1f}%，严重度 {sev_cn}。")
    if action == "PASS":
        tail = "处置：工件无缺陷，直接放行，无需机械臂动作。"
    else:
        act_cn = "分拣" if action == "SORT" else "转人工复检"
        tail = f"处置：执行【{act_cn}】" + (f" → 投放至「{dest_label}」。" if dest_label else "。")
    advice = rule_advice or DEFECT_RULES.get(cls_en, {}).get("advice", "")
    mid = f"判据：{dec.get('reason', '')}" if dec.get("reason") else ""
    parts = [head, mid, (f"工艺建议：{advice}。" if advice else ""), tail]
    return "\n".join(p for p in parts if p)


def _strip_think(text: str) -> str:
    """剥离 deepseek-r1 的思考块。"""
    if not text:
        return ""
    text = re.sub(r"<think[\s\S]*?</think>", "", text, flags=re.I)
    text = re.sub(r"<think[\s\S]*$", "", text, flags=re.I)
    return text.strip()


def _extract_json(text: str) -> Optional[dict]:
    """从模型输出里抠出第一个完整 JSON 对象（容忍代码块和多余文字）。"""
    cleaned = _strip_think(text)
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = cleaned[start:end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        # 再宽松一点：删掉尾随逗号
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", candidate))
        except Exception:
            return None


def ollama_available(timeout: float = 3.0) -> tuple[bool, list[str]]:
    """探测 Ollama 是否在线，返回 (是否在线, 模型名列表)。"""
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=timeout)
        if r.status_code == 200:
            return True, [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:
        pass
    return False, []


def resolve_model(preferred: str = "", timeout: float = 3.0) -> str:
    """
    挑一个本机真实存在的模型：优先 preferred / OLLAMA_MODEL，
    其次 OLLAMA_FALLBACK_MODELS，最后用本机任意一个模型。
    返回空字符串表示 Ollama 不可用或没有任何模型。
    """
    ok, models = ollama_available(timeout=timeout)
    if not ok or not models:
        return ""
    for want in [preferred, OLLAMA_MODEL, *OLLAMA_FALLBACK_MODELS]:
        if want and want in models:
            return want
    # 前缀匹配（ollama 的 name 可能带 :latest）
    for want in [preferred, OLLAMA_MODEL]:
        if want:
            for m in models:
                if m.split(":")[0] == want.split(":")[0]:
                    return m
    return models[0]


def ollama_chat(prompt: str, system: str = "", model: str = "", timeout: float = OLLAMA_TIMEOUT,
                json_format: bool = False) -> Optional[str]:
    """同步调用 Ollama /api/chat；失败返回 None（调用方负责降级）。"""
    model = model or resolve_model()
    if not model:
        print("[LLM] 没有可用模型（Ollama 未启动或未 pull 模型），走规则决策")
        return None
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1},
    }
    if json_format:
        payload["format"] = "json"
    try:
        r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=timeout)
        if r.status_code == 200:
            return r.json().get("message", {}).get("content")
        print(f"[LLM] Ollama HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"[LLM] Ollama 调用失败: {e}")
    return None


def llm_decide(detect: dict, base: dict, model: str = "", timeout: float = OLLAMA_TIMEOUT) -> Optional[dict]:
    """让大模型基于检测结果 + 规则底线重新组织语言和严重度。失败返回 None。"""
    prompt = (
        "视觉检测结果：\n"
        f"- 是否存在缺陷: {detect.get('has_defect')}\n"
        f"- 缺陷类别: {detect.get('defect_cls') or '无'}（中文: {defect_cn(detect.get('defect_cls') or '')}）\n"
        f"- 置信度: {detect.get('conf')}\n"
        f"- 缺陷框(像素 xyxy): {detect.get('bbox_xyxy')}\n"
        f"- 工件世界坐标(米): {detect.get('world_xy')}\n"
        f"- 规则底线结论: {base.get('action')} / {base.get('dest_label') or '放行'}\n\n"
        "请给出你的处置决策 JSON。"
    )
    raw = ollama_chat(prompt, system=_SYS_PROMPT, model=model, timeout=timeout, json_format=True)
    if not raw:
        return None
    plan = _extract_json(raw)
    if not isinstance(plan, dict) or "action" not in plan:
        return None

    # 用模型结果覆盖"表述"，但保留规则脑的兜底字段，避免模型给出非法坑位
    dest = str(plan.get("dest") or base.get("dest") or "")
    if dest not in DEST_BINS and dest != "":
        dest = base.get("dest", "review")
    action = str(plan.get("action") or base.get("action", "REVIEW")).upper()
    if action not in ("SORT", "REVIEW", "PASS"):
        action = base.get("action", "REVIEW")
    if action == "PASS":
        dest = ""
    severity = str(plan.get("severity") or base.get("severity", "medium")).lower()
    if severity not in _SEVERITY_RANK:
        severity = base.get("severity", "medium")

    label = DEST_BINS.get(dest, {}).get("label", "")
    out = {
        "action": action,
        "dest": dest,
        "dest_label": label,
        "reason": str(plan.get("reason") or base.get("reason", ""))[:120],
        "severity": severity,
        "engine": f"ollama:{model or OLLAMA_MODEL}",
    }
    out["report"] = _compose_report(detect, out)
    return out


def decide(detect: dict, use_llm: bool = True, model: str = "",
           timeout: float = OLLAMA_TIMEOUT) -> dict:
    """
    对外唯一入口：给定 YOLO 检测结果，返回最终决策。
    use_llm=False 时只用规则脑（界面上的"跳过 LLM"开关 / 演示降级）。
    """
    base = rule_decide(detect)
    if not use_llm:
        return base

    t0 = time.time()
    llm = llm_decide(detect, base, model=model, timeout=timeout)
    if llm is None:
        base["llm_ms"] = int((time.time() - t0) * 1000)
        base["llm_ok"] = False
        return base

    llm["llm_ms"] = int((time.time() - t0) * 1000)
    llm["llm_ok"] = True
    # 大模型没给出关键结论时回填规则脑，保证不空
    for k in ("reason", "report", "dest_label"):
        if not llm.get(k):
            llm[k] = base.get(k, "")
    return llm


if __name__ == "__main__":
    demo = [
        {"has_defect": True, "defect_cls": "scratches", "conf": 0.81,
         "bbox_xyxy": [120, 80, 260, 190], "world_xy": [0.05, -0.03]},
        {"has_defect": True, "defect_cls": "crazing", "conf": 0.44,
         "bbox_xyxy": [30, 40, 90, 120], "world_xy": [-0.12, 0.06]},
        {"has_defect": False, "defect_cls": "", "conf": 0.0,
         "bbox_xyxy": [], "world_xy": None},
    ]
    print("Ollama:", ollama_available())
    for d in demo:
        dec = decide(d)
        print("\n检测:", d.get("defect_cls"), d.get("conf"))
        print(json.dumps(dec, ensure_ascii=False, indent=2))
