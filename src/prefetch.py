"""Study 级预解码：后台线程读图解码，主线程做预处理与 GPU 前向。

数值与串行完全一致（同一个 load_letterbox、同一个线程池配置），只改变调度：
解码下一个 Study 的同时计算当前 Study，端到端 ≈ max(CPU, GPU) 而不是两者相加。

队列深度 2 就够：再深只多占内存，不会让 GPU 更饱和。
"""
from __future__ import annotations

import queue
import threading
import time


def prefetch_studies(studies, decode, qsize: int = 2, timing: dict | None = None):
    """按 Study 预解码。

    studies  可迭代，元素为 (sid, paths, rows)，原样透传
    decode   paths -> np.ndarray，在后台线程里调用
    timing   可选，累加 "decode_wait"（主线程等队列的秒数）
    产出     (sid, paths, rows, images)
    """
    q: queue.Queue = queue.Queue(maxsize=qsize)
    done = object()

    def produce():
        try:
            for sid, paths, rows in studies:
                q.put((sid, paths, rows, decode(paths)))
        except BaseException as exc:      # 解码异常必须回传到主线程，不能吞掉
            q.put(exc)
        finally:
            q.put(done)

    th = threading.Thread(target=produce, name="prefetch", daemon=True)
    th.start()
    while True:
        t0 = time.perf_counter()
        item = q.get()
        if timing is not None:
            timing["decode_wait"] = timing.get("decode_wait", 0.0) + (time.perf_counter() - t0)
        if item is done:
            break
        if isinstance(item, BaseException):
            raise item
        yield item
    th.join()


def iter_studies(studies, decode, use_prefetch: bool = True,
                 qsize: int = 2, timing: dict | None = None):
    """统一入口：返回 (sid, paths, rows, images)。

    串行分支把解码时间记进 timing["decode"]，预解码分支记的是
    timing["decode_wait"]（只统计没被 GPU 藏住的那部分）。
    """
    if use_prefetch:
        yield from prefetch_studies(studies, decode, qsize=qsize, timing=timing)
        return
    for sid, paths, rows in studies:
        t0 = time.perf_counter()
        images = decode(paths)
        if timing is not None:
            timing["decode"] = timing.get("decode", 0.0) + (time.perf_counter() - t0)
        yield sid, paths, rows, images
