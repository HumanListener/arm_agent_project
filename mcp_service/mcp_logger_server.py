"""
巡检日志 MCP 服务。
- 底层 SQLite 持久化每一轮巡检记录；
- 用 FastMCP 把 write_inspection_log / query_recent_logs 暴露成 MCP 工具（体现 MCP 协议）；
- Agent 进程内直接 import InspectionLogger 即可调用，无需单独起进程；
- 也可以独立运行：python -m mcp_service.mcp_logger_server 启动 MCP server 供外部客户端连接。
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.settings import DB_PATH

_lock = threading.Lock()


def _init_db(db_path: Path = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS inspection_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            grid_index INTEGER,
            has_defect INTEGER,
            defect_cls TEXT,
            conf REAL,
            world_xy TEXT,
            llm_action TEXT,
            detail TEXT
        )
    """)
    conn.commit()
    conn.close()


class InspectionLogger:
    """进程内日志器：Agent 直接调它；同时被注册为 MCP 工具。"""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        _init_db(self.db_path)

    # ---------- SQLite 操作 ----------
    def write_log(self, grid_index: int, has_defect: bool, defect_cls: str,
                  conf: float, world_xy: Optional[list], llm_action: str,
                  detail: str = "") -> int:
        with _lock, sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO inspection_logs(ts, grid_index, has_defect, defect_cls, conf, world_xy, llm_action, detail)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), grid_index,
                 int(has_defect), defect_cls, conf,
                 json.dumps(world_xy) if world_xy else None,
                 llm_action, detail),
            )
            conn.commit()
            return cur.lastrowid

    def query_recent(self, limit: int = 20) -> list[dict]:
        with _lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM inspection_logs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]


# ---------- MCP 工具注册（FastMCP）----------
# 单独启动：python -m mcp_service.mcp_logger_server
try:
    from mcp.server.fastmcp import FastMCP

    mcp_server = FastMCP("inspection-logger")
    _logger = InspectionLogger()

    @mcp_server.tool()
    def write_inspection_log(grid_index: int, has_defect: bool, defect_cls: str,
                             conf: float, world_xy: Optional[list],
                             llm_action: str, detail: str = "") -> int:
        """写入一条钢材巡检日志。"""
        return _logger.write_log(grid_index, has_defect, defect_cls, conf,
                                 world_xy, llm_action, detail)

    @mcp_server.tool()
    def query_recent_logs(limit: int = 20) -> list[dict]:
        """查询最近 N 条巡检日志。"""
        return _logger.query_recent(limit)

    def run_server():
        print("[MCP] inspection-logger server 启动（stdio 模式）")
        mcp_server.run()

except Exception as e:  # mcp 版本差异时不影响进程内调用
    print(f"[MCP] FastMCP 注册跳过: {e}")
    mcp_server = None
    _logger = InspectionLogger()

    def run_server():
        raise RuntimeError("当前环境未安装 mcp，无法启动独立 MCP server")


# Agent 直接用这个单例
logger = InspectionLogger()


if __name__ == "__main__":
    run_server()
