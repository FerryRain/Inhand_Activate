#!/usr/bin/env python3
"""Batch geometry-only inference with the released TRELLIS.2-4B model."""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import trimesh
from PIL import Image


TRELLIS_ROOT = Path("/home/ferry/data/Code2/Research/TRELLIS/TRELLIS.2")
sys.path.insert(0, str(TRELLIS_ROOT))
from trellis2.pipelines import Trellis2ImageTo3DPipeline  # noqa: E402


def remove_voxel_aabb_shell(mesh: trimesh.Trimesh, threshold: float = 0.45):
    """Remove decoder-domain faces lying on the six +/-0.5 AABB planes.

    These faces are not part of the predicted object, but can appear when an
    O-voxel level set reaches the finite decoder boundary.  Filtering is done
    face-wise and leaves central predicted geometry untouched.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangle_vertices = vertices[faces]
    on_positive = np.all(triangle_vertices > float(threshold), axis=1)
    on_negative = np.all(triangle_vertices < -float(threshold), axis=1)
    on_boundary_plane = np.any(on_positive | on_negative, axis=1)
    if np.count_nonzero(on_boundary_plane):
        mesh.update_faces(~on_boundary_plane)
        mesh.remove_unreferenced_vertices()
    return mesh, int(np.count_nonzero(on_boundary_plane))


def remove_decoder_shell_gpu(result, threshold: float = 0.45):
    triangles = result.vertices[result.faces.long()]
    on_positive = torch.all(triangles > float(threshold), dim=1)
    on_negative = torch.all(triangles < -float(threshold), dim=1)
    boundary = torch.any(on_positive | on_negative, dim=1)
    removed = int(boundary.sum().item())
    if removed:
        result.remove_faces(boundary)
    return removed


@torch.no_grad()
def run_geometry_only(pipeline, image, seed: int = 42, max_num_tokens: int = 49152):
    """Run the released TRELLIS.2 shape path without sampling texture latent.

    The official ``pipeline.run`` samples shape first and texture second, then
    combines both for a textured asset.  Downstream evaluation only consumes
    vertices/faces, so decoding the identical shape latent directly avoids the
    unrelated texture model and its VRAM fragmentation.
    """
    pipeline_type = pipeline.default_pipeline_type
    image = pipeline.preprocess_image(image)
    torch.manual_seed(int(seed))
    cond_512 = pipeline.get_cond([image], 512)
    cond_1024 = pipeline.get_cond([image], 1024) if pipeline_type != "512" else None
    sparse_resolution = {
        "512": 32,
        "1024": 64,
        "1024_cascade": 32,
        "1536_cascade": 32,
    }[pipeline_type]
    coords = pipeline.sample_sparse_structure(cond_512, sparse_resolution, 1, {})
    if pipeline_type == "512":
        shape_slat = pipeline.sample_shape_slat(
            cond_512, pipeline.models["shape_slat_flow_model_512"], coords, {}
        )
        resolution = 512
    elif pipeline_type == "1024":
        shape_slat = pipeline.sample_shape_slat(
            cond_1024, pipeline.models["shape_slat_flow_model_1024"], coords, {}
        )
        resolution = 1024
    elif pipeline_type in ("1024_cascade", "1536_cascade"):
        high_resolution = 1024 if pipeline_type == "1024_cascade" else 1536
        shape_slat, resolution = pipeline.sample_shape_slat_cascade(
            cond_512,
            cond_1024,
            pipeline.models["shape_slat_flow_model_512"],
            pipeline.models["shape_slat_flow_model_1024"],
            512,
            high_resolution,
            coords,
            {},
            max_num_tokens,
        )
    else:
        raise ValueError(f"Unsupported TRELLIS.2 pipeline type: {pipeline_type}")
    meshes, _ = pipeline.decode_shape_slat(shape_slat, resolution)
    for mesh in meshes:
        mesh.fill_holes()
    torch.cuda.empty_cache()
    return meshes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="rebuttal/assets/ycb/single_view_inputs")
    parser.add_argument(
        "--output", default="rebuttal/results/downstream_single_view_raw/trellis2"
    )
    parser.add_argument(
        "--model",
        default="/home/ferry/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B",
    )
    parser.add_argument("--decimation-target", type=int, default=200000)
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument(
        "--max-num-tokens",
        type=int,
        default=49152,
        help="Official cascade token cap; lower uniformly on memory-limited GPUs.",
    )
    parser.add_argument(
        "--official-glb-postprocess",
        action="store_true",
        help="Also run textured GLB remeshing; geometry-only evaluation does not require it.",
    )
    parser.add_argument(
        "--reload-every",
        type=int,
        default=1,
        help="Reload the 4B pipeline after this many new meshes to bound VRAM fragmentation.",
    )
    parser.add_argument(
        "--max-new",
        type=int,
        default=0,
        help="Stop after this many newly generated meshes (0 processes all pending inputs).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("TRELLIS.2 downstream inference requires CUDA")
    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    inputs = sorted(input_root.glob("*/pose_*.png"))
    if not inputs:
        raise RuntimeError(f"No rendered inputs under {input_root}")
    pipeline = None
    processed = 0
    for index, image_path in enumerate(inputs):
        object_name = image_path.parent.name
        output = output_root / object_name / f"{image_path.stem}.ply"
        glb_output = output.with_suffix(".glb")
        if output.exists() and not args.overwrite:
            continue
        if pipeline is None:
            pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
            pipeline.cuda()
        image = Image.open(image_path).convert("RGBA")
        with torch.inference_mode():
            result = run_geometry_only(
                pipeline, image, max_num_tokens=int(args.max_num_tokens)
            )[0]
        output.parent.mkdir(parents=True, exist_ok=True)
        removed_shell_faces = remove_decoder_shell_gpu(result)
        if len(result.faces) > args.decimation_target:
            result.simplify(args.decimation_target)
        if args.official_glb_postprocess:
            import o_voxel

            glb = o_voxel.postprocess.to_glb(
                vertices=result.vertices,
                faces=result.faces,
                attr_volume=result.attrs,
                coords=result.coords,
                attr_layout=result.layout,
                voxel_size=result.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=args.decimation_target,
                texture_size=args.texture_size,
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                verbose=False,
            )
            glb.export(glb_output)
            loaded = trimesh.load(glb_output, force="scene", process=True)
            if isinstance(loaded, trimesh.Scene):
                if not loaded.geometry:
                    raise RuntimeError(f"TRELLIS.2 exported an empty scene: {glb_output}")
                mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
            else:
                mesh = loaded
            del glb, loaded
        else:
            vertices = result.vertices.detach().float().cpu().numpy()
            faces = result.faces.detach().long().cpu().numpy()
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        if len(mesh.faces) < 100:
            raise RuntimeError(
                f"TRELLIS.2 object surface vanished after AABB cleanup: {glb_output}"
            )
        mesh.remove_unreferenced_vertices()
        mesh.export(output)
        print(
            f"[{index + 1}/{len(inputs)}] {output} "
            f"(removed {removed_shell_faces} decoder-boundary faces)",
            flush=True,
        )
        del result, mesh
        torch.cuda.empty_cache()
        processed += 1
        if args.max_new > 0 and processed >= args.max_new:
            break
        if args.reload_every > 0 and processed % args.reload_every == 0:
            del pipeline
            pipeline = None
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
