"""Predict chest X-ray probabilities from user-supplied local images and metadata."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def validate_inputs(meta, studies, smoothing):
    seen_images = [image_id for _, _, rows in studies for image_id in rows["Dicom_id"]]
    if len(seen_images) != len(set(seen_images)):
        raise ValueError("Image filenames must have unique stems")
    selected_studies = {sid for sid, _, _ in studies}
    selected = meta[meta["Study_id"].isin(selected_studies)]
    if smoothing:
        selected = meta[meta["Subject_id"].isin(selected["Subject_id"])]
    if set(seen_images) != set(selected["Dicom_id"]):
        raise ValueError("Provide every image of each selected study, and every listed study for each patient when smoothing")


def load_image(path):
    from src.prepare_cache import load_gray, preprocess

    if Path(path).suffix.lower() in {".jpg", ".jpeg"}:
        import numpy as np
        import torch
        from torchvision.io import ImageReadMode, decode_jpeg

        encoded = np.fromfile(path, dtype=np.uint8)
        if not encoded.size:
            raise ValueError(f"Empty image file: {path}")
        image = decode_jpeg(torch.from_numpy(encoded), mode=ImageReadMode.GRAY,
                            device="cpu", apply_exif_orientation=False).squeeze(0).numpy()
    else:
        image = load_gray(path)
    return preprocess(image, 448)


def setup_cuda_graph(ensemble):
    import cv2
    import torch
    import torch._dynamo.config as dynamo_config
    from triton.runtime import driver
    from src.cuda_graph import CompiledRunner, graph_predict_study

    driver.active.get_current_target()
    cv2.setNumThreads(8)
    dynamo_config.recompile_limit = 64
    dynamo_config.accumulated_recompile_limit = 256
    dynamo_config.suppress_errors = False
    dynamo_config.fail_on_recompile_limit_hit = True
    ensemble.graph_runner = CompiledRunner(ensemble.members, ensemble.device)
    ensemble.graph_runner.prepare()
    return graph_predict_study


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pictures", required=True, type=Path)
    parser.add_argument("--metadata", "--meta-csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--deploy", type=Path,
                        default=Path(__file__).resolve().parent / "configs/ensemble.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--backend", choices=("eager", "cuda-graph"), default="eager")
    parser.add_argument("--decode-threads", type=int, default=4)
    parser.add_argument("--no-patient-smoothing", action="store_true")
    args = parser.parse_args(argv)
    if args.decode_threads < 1:
        parser.error("--decode-threads must be positive")
    if args.out.exists():
        parser.error("Output exists; choose a new path")
    if args.backend == "cuda-graph":
        if sys.platform != "linux" or not shutil.which("g++"):
            parser.error("CUDA Graph requires Linux and g++")
        if not (Path(sysconfig.get_path("include")) / "Python.h").is_file():
            parser.error("CUDA Graph requires Python development headers")
        os.environ["TORCHINDUCTOR_EMULATE_PRECISION_CASTS"] = "1"
        os.environ["TORCHINDUCTOR_EMULATE_DIVISION_ROUNDING"] = "1"
        os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")

    import numpy as np
    import pandas as pd
    import torch
    from src.aggregate import parse_patient_smooth, patient_smooth
    from src.predict import Ensemble, collect_studies, load_deploy, predict_study
    from src.submit import build_submission, write_submission

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.backend == "cuda-graph" and (device != "cuda" or torch.__version__.split("+")[0] != "2.10.0"):
        parser.error("CUDA Graph requires PyTorch 2.10.0 and a CUDA GPU")

    deploy = load_deploy(args.deploy)
    smoothing = None if args.no_patient_smoothing else parse_patient_smooth(deploy.get("patient_smooth"))
    keys = ["Dicom_id", "Study_id", "Subject_id"]
    required = set(keys + ["Rows", "Columns", "ViewPosition", "PatientOrientation"])
    meta = pd.read_csv(args.metadata, dtype={key: str for key in keys})
    missing = required - set(meta)
    if missing:
        parser.error(f"Missing metadata columns: {', '.join(sorted(missing))}")
    if (meta[keys].isna().any().any() or meta["Dicom_id"].duplicated().any()
            or (meta.groupby("Study_id")["Subject_id"].nunique() != 1).any()):
        parser.error("Each image needs a unique ID and each study needs a single patient ID")
    for column in ("Rows", "Columns"):
        values = pd.to_numeric(meta[column], errors="coerce")
        if not (np.isfinite(values) & (values > 0)).all():
            parser.error(f"{column} must contain positive image dimensions")
        meta[column] = values
    studies = collect_studies(args.pictures, meta)
    validate_inputs(meta, studies, smoothing is not None)
    norm = json.loads(Path(deploy["norm_json"]).read_text(encoding="utf-8"))
    if args.backend == "cuda-graph" and any(len(paths) > 8 for _, paths, _ in studies):
        parser.error("CUDA Graph supports at most eight images per study; choose eager for larger studies")

    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    ensemble = Ensemble(deploy["ckpts"], deploy["weights"], device, deploy["precision"])
    predict_one = setup_cuda_graph(ensemble) if args.backend == "cuda-graph" else predict_study
    study_ids, subjects, probabilities = [], [], []
    with ThreadPoolExecutor(max_workers=args.decode_threads) as pool:
        for study_id, paths, rows in studies:
            images = np.stack(list(pool.map(load_image, paths)))
            probabilities.append(predict_one(ensemble, images, rows, norm))
            study_ids.append(study_id)
            subjects.append(rows["Subject_id"].iloc[0])
    probabilities = np.stack(probabilities)
    if smoothing is not None:
        probabilities = patient_smooth(probabilities, np.array(subjects), smoothing)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_submission(build_submission(study_ids, probabilities), args.out, expected_study_ids=study_ids)
    print(f"Wrote ten probabilities per study for {len(study_ids)} studies: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
