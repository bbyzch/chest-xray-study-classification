"""CUDA Graph inference and input sharing for the four trained students."""
from __future__ import annotations

import json
import time

import numpy as np
import torch

FEEDS = ("image", "view_idx", "orient_idx", "size_feat")
MAX_IMAGES = 8



COMPILED_NAMES = frozenset({
    "convnext_base_s384_seed0_pre22k-distill0.6t2-bnorm_full_stop10",
    "convnext_base_s448_seed0_pre22k-distill0.6t2-bnorm_full_stop8",
    "convnext_tiny_s384_seed0_distill0.6t2-nocl_full_stop8",
})
NATIVE_NAMES = frozenset({
    "swin_v2_t_s384_seed0_distill0.6t2-e24_full_stop14",
})


def prepare_inputs(ens, images, rows, norm, cache_size):
    from torchvision.transforms.v2 import functional as F
    from src.dataset import _get_clahe

    if len(images) != len(rows) or not len(rows):
        raise ValueError("Nonempty image and metadata counts must match")
    ids = np.asarray(rows["row_idx"], dtype=np.int64)
    if np.any(ids < 0) or np.any(ids >= len(images)):
        raise ValueError("Invalid local image index")
    log_h = np.log(np.clip(np.asarray(rows["Rows"], dtype=np.float64), 1, None))
    log_w = np.log(np.clip(np.asarray(rows["Columns"], dtype=np.float64), 1, None))
    sm, ss = norm["size_mean"], norm["size_std"]
    size_feat = np.stack([(log_h - sm[0]) / ss[0],
                          (log_w - sm[1]) / ss[1]], axis=1).astype(np.float32)
    metadata = {
        "view_idx": torch.from_numpy(np.array(rows["view_idx"], dtype=np.int64, copy=True)).to(ens.device),
        "orient_idx": torch.from_numpy(np.array(rows["orient_idx"], dtype=np.int64, copy=True)).to(ens.device),
        "size_feat": torch.from_numpy(size_feat).to(ens.device),
    }
    keys = list(dict.fromkeys((m["image_size"], m["use_clahe"]) for m in ens.members))
    variants = {}
    for clahe in dict.fromkeys(key[1] for key in keys):
        tensors = []
        for idx in ids:
            img = np.array(images[int(idx)], dtype=np.uint8, copy=True)
            if clahe:
                img = _get_clahe().apply(img)
            t = torch.from_numpy(img).unsqueeze(0)
            size = cache_size or max(t.shape[-2:])
            if size != t.shape[-1]:
                t = F.resize(t, [size, size], antialias=True)
            tensors.append(t)
        variants[clahe] = tensors
    inputs = {}
    for size, clahe in keys:
        normalized = []
        for t in variants[clahe]:
            t = F.resize(t, [size, size], antialias=True)
            # Preserve uint8 resize and the original separate float32 operations.
            t = t.to(torch.float32).div_(255.0)
            normalized.append(t.sub_(float(norm["mean"])).div_(float(norm["std"])))
        inputs[(size, clahe)] = {"image": torch.stack(normalized).to(ens.device), **metadata}
    return inputs


@torch.no_grad()
def graph_predict_study(ens, images, rows, norm, cache_size=448, timing=None):
    if len(images) != len(rows):
        raise ValueError(f"Image count {len(images)} differs from metadata count {len(rows)}")
    t0 = time.perf_counter()
    inputs = prepare_inputs(ens, images, rows, norm, cache_size)
    t1 = time.perf_counter()
    logits = ens.graph_runner.run(inputs)
    # Preserve the original operation order and CPU float64 ensemble arithmetic.
    probs = [torch.sigmoid(z.float()) for z in logits]
    probs = [p.cpu().numpy().astype(np.float64) for p in probs]
    t2 = time.perf_counter()
    total = None
    for member, p in zip(ens.members, probs):
        p = p * member["weight"]
        total = p if total is None else total + p
    if timing is not None:
        timing["preprocess"] = timing.get("preprocess", 0.0) + (t1 - t0)
        timing["gpu"] = timing.get("gpu", 0.0) + (t2 - t1)
    return total.mean(axis=0)


class JointGraphRunner:
    def __init__(self, members, device):
        self.members = members
        self.device = torch.device(device)
        self.states = {}
        self.summary = {"fixed_batch_sizes": list(range(1, MAX_IMAGES + 1)),
                        "shared_pool_across_batch_graphs": True,
                        "graph_scope": "Four forwards only; sigmoid and CPU aggregation outside",
                        "autocast_cache_enabled_during_capture": False}

    def _amp(self, cache_enabled=False):
        return torch.autocast(device_type="cuda", dtype=torch.float16,
                              enabled=True, cache_enabled=cache_enabled)

    def _forward(self, static_inputs):
        return tuple(m["model"](*(values[k] for k in FEEDS))
                     for m, values in zip(self.members, static_inputs))

    @torch.no_grad()
    def prepare(self):
        if self.states:
            return
        if len(self.members) != 4:
            raise ValueError("Expected the original four members")
        t0 = time.perf_counter()
        with torch.cuda.device(self.device):
            current = torch.cuda.current_stream(self.device)
            side = torch.cuda.Stream(device=self.device)
            side.wait_stream(current)
            pool = torch.cuda.graph_pool_handle()
            self.pool = pool
            self.capture_stream = side
            for batch in range(1, MAX_IMAGES + 1):
                # Buffers and graph outputs are kept alive for the runner lifetime.
                with torch.cuda.stream(side):
                    static_inputs = []
                    for member in self.members:
                        size = member["image_size"]
                        static_inputs.append({
                            "image": torch.zeros(batch, 1, size, size, device=self.device),
                            "view_idx": torch.zeros(batch, dtype=torch.long, device=self.device),
                            "orient_idx": torch.zeros(batch, dtype=torch.long, device=self.device),
                            "size_feat": torch.zeros(batch, 2, device=self.device),
                        })
                    # cuDNN/cuBLAS first-use initialization happens on a side stream.
                    with self._amp(cache_enabled=False):
                        for _ in range(3):
                            warm_outputs = self._forward(static_inputs)
                            del warm_outputs
                side.synchronize()
                graph = torch.cuda.CUDAGraph()
                # These graphs are independent and replay serially; all outputs
                # are consumed before the next replay, so sharing is safe.
                # All captures use the same stream for effective pool sharing.
                with torch.cuda.graph(graph, pool=pool, stream=side):
                    with self._amp(cache_enabled=False):
                        outputs = self._forward(static_inputs)
                self.states[batch] = {"inputs": static_inputs, "outputs": outputs, "graph": graph}
                current.wait_stream(side)
                print(f"[GRAPH] Captured fixed four-member graph, batch={batch}", flush=True)

            current.wait_stream(side)
            torch.cuda.synchronize(self.device)
            probe_errors = []
            # Initialization-only synthetic probes. No test-image data is read here.
            # Every graph's outputs are consumed on CPU before another graph replay.
            for batch, state in self.states.items():
                for i, values in enumerate(state["inputs"]):
                    values["image"].fill_(0.25 + 0.01 * i)
                    values["size_feat"].fill_(-0.2)
                # Match ordinary eager AMP semantics for the native reference.
                with self._amp(cache_enabled=True):
                    native = self._forward(state["inputs"])
                native_cpu = [torch.sigmoid(t.float()).cpu().numpy() for t in native]
                del native
                state["graph"].replay()
                graph_cpu = [torch.sigmoid(t.float()).cpu().numpy() for t in state["outputs"]]
                error = max(float(np.abs(a - b).max()) for a, b in zip(native_cpu, graph_cpu))
                if not np.isfinite(error) or error > 1e-3:
                    raise RuntimeError(f"Graph initialization disagrees with eager AMP: batch={batch}, max_delta={error}")
                probe_errors.append({"batch": batch, "max_probability_error_vs_eager": error})
            torch.cuda.synchronize(self.device)
            self.summary.update(initialization_seconds=time.perf_counter() - t0,
                                synthetic_probe_errors=probe_errors,
                                peak_allocated_MiB=torch.cuda.max_memory_allocated(self.device) / 2**20,
                                reserved_MiB=torch.cuda.memory_reserved(self.device) / 2**20)
        print("[GRAPH] Initialization summary: " + json.dumps(self.summary), flush=True)

    @torch.no_grad()
    def run(self, inputs):
        if not inputs:
            raise ValueError("No preprocessed inputs")
        batch = next(iter(inputs.values()))["image"].shape[0]
        if batch not in self.states:
            raise ValueError(f"Batch {batch} was not initialized in the fixed 1..{MAX_IMAGES} range; no timed capture/fallback")
        state = self.states[batch]
        for member, target in zip(self.members, state["inputs"]):
            actual = inputs[(member["image_size"], member["use_clahe"])]
            for name in FEEDS:
                src, dst = actual[name], target[name]
                if src.shape != dst.shape or src.dtype != dst.dtype or src.device != dst.device:
                    raise ValueError(f"Graph input mismatch for {name}: {src.shape}/{src.dtype}/{src.device}")
                dst.copy_(src)
        state["graph"].replay()
        # These buffers will be reused. Caller must finish sigmoid/CPU extraction
        # before the next Study, as graph_predict_study above does synchronously.
        return state["outputs"]


def select_compiled(members):
    names = [m["name"] for m in members]
    if len(names) != 4 or set(names) != COMPILED_NAMES | NATIVE_NAMES:
        raise ValueError(f"Expected the original fixed four-member check package, got {names}")
    return [m for m in members if m["name"] in COMPILED_NAMES]


class CompiledRunner(JointGraphRunner):
    def __init__(self, members, device):
        super().__init__(members, device)
        self.original_models = tuple(m["model"] for m in members)
        selected = select_compiled(members)
        for member in selected:
            member["model"] = torch.compile(
                member["model"], backend="inductor", fullgraph=True,
                dynamic=False, options={"triton.cudagraphs": False})
            print(f"[COMPILE] Prepared {member['name']}; actual compilation occurs during fixed-shape warmup", flush=True)
        self.summary.update(compiled_members=[m["name"] for m in selected],
                            compiler="Inductor default, static shapes, fullgraph",
                            compiler_internal_cuda_graphs=False,
                            native_members=[m["name"] for m in members if m["name"] in NATIVE_NAMES],
                            emulate_precision_casts=True, emulate_division_rounding=True)

    @torch.no_grad()
    def prepare(self):
        super().prepare()
        import numpy as np
        checks = []
        # Compare original uncompiled models with compiled captured models.
        # Synthetic inputs only; this is additional initialization, not a
        # replacement for real-image CSV probability/ranking verification.
        for batch, state in self.states.items():
            with self._amp(cache_enabled=False):
                eager = tuple(model(*(values[k] for k in FEEDS))
                              for model, values in zip(self.original_models, state["inputs"]))
            ref = [torch.sigmoid(t.float()).cpu().numpy() for t in eager]
            del eager
            state["graph"].replay()
            got = [torch.sigmoid(t.float()).cpu().numpy() for t in state["outputs"]]
            delta = max(float(np.abs(a - b).max()) for a, b in zip(ref, got))
            checks.append({"batch": batch, "max_probability_error_vs_original": delta})
            if not np.isfinite(delta) or delta > 1e-3:
                self.summary["original_model_probe_checks"] = checks
                raise RuntimeError(f"Compiled model differs from original: batch={batch}, delta={delta}")
        self.summary["original_model_probe_checks"] = checks
        print("[COMPILE] Original model probe: " + json.dumps(checks), flush=True)
