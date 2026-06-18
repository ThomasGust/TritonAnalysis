import json
from pathlib import Path

import cv2
import numpy as np
from pytest import approx

import triton_analysis.realityscan.underwater_pipeline as pipeline
from triton_analysis.realityscan.underwater_pipeline import (
    DEFAULT_FAST_DISTORTION_MODEL,
    DEFAULT_FAST_GEOMETRY_MODE,
    DEFAULT_MIN_GOOD_COMPONENT_RATIO,
    LEGACY_DISTORTION_MODEL,
    LEGACY_GEOMETRY_MODE,
    MISSING_REALITYSCAN_OUTPUT_EXIT_CODE,
    FrameMetric,
    VariantSpec,
    apply_legacy_enhanced_default,
    apply_reconstruction_preset,
    build_arg_parser,
    build_variant_specs,
    filter_obj_large_faces,
    final_reconstruction_exit_code,
    estimate_caustic_score,
    load_stereo_calibration,
    load_stereo_session,
    make_geometry_frame,
    make_variant_paths,
    parse_alignment_report,
    prepare_output_paths,
    read_stereo_session_metrics,
    run_alignment_tournament,
    scale_model_from_stereo_baseline,
    selected_image_prior_commands,
    select_frames,
    write_alignment_results,
    write_realityscan_command_file,
    write_stereo_variant_frames,
)


def _write_image(path: Path, offset: int) -> None:
    y, x = np.indices((12, 16))
    image = np.dstack(
        [
            (x * 12 + offset) % 255,
            (y * 18 + offset) % 255,
            ((x + y) * 9 + offset) % 255,
        ]
    ).astype(np.uint8)
    assert cv2.imwrite(str(path), image)


def _make_stereo_session(tmp_path: Path) -> Path:
    session_dir = tmp_path / "stereo_sessions" / "test-session"
    (session_dir / "left").mkdir(parents=True)
    (session_dir / "right").mkdir()
    frames = []
    for index in range(1, 3):
        stem = f"pair_{index:06d}"
        left = session_dir / "left" / f"{stem}_left.png"
        right = session_dir / "right" / f"{stem}_right.png"
        _write_image(left, index)
        _write_image(right, index + 20)
        frames.append(
            {
                "index": index,
                "stem": stem,
                "left_path": f"left\\{stem}_left.png",
                "right_path": f"right\\{stem}_right.png",
                "pair_delta_ms": 1.5,
                "left": {"wall_ts": 100.0 + index, "shape": [12, 16, 3]},
                "right": {"wall_ts": 100.0015 + index, "shape": [12, 16, 3]},
            }
        )
    (session_dir / "manifest.json").write_text(json.dumps({"frames": frames}), encoding="utf-8")
    return session_dir


def _make_calibration(tmp_path: Path, *, units: str | None = None, baseline: float = 100.0) -> Path:
    calibration = {
        "image_size": [16, 12],
        "rig_id": "unit_test_rig",
        "left": {
            "camera_matrix": [[10.0, 0.0, 8.0], [0.0, 11.0, 6.0], [0.0, 0.0, 1.0]],
            "dist_coeffs": [[0.1, 0.01, 0.001, 0.002, 0.0001]],
        },
        "right": {
            "camera_matrix": [[10.5, 0.0, 7.5], [0.0, 10.8, 6.5], [0.0, 0.0, 1.0]],
            "dist_coeffs": [[0.2, 0.02, 0.003, 0.004, 0.0002]],
        },
        "stereo": {
            "baseline": baseline,
            "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "translation": [-baseline, 0.0, 0.0],
        },
    }
    if units is not None:
        calibration["board"] = {"units": units}
    path = tmp_path / f"stereo_calibration_{units or 'default'}.json"
    path.write_text(json.dumps(calibration), encoding="utf-8")
    return path


def test_load_stereo_session_scores_pairs(tmp_path: Path):
    session = load_stereo_session(_make_stereo_session(tmp_path))

    info, metrics = read_stereo_session_metrics(session, max_pair_delta_ms=75.0)

    assert info.frame_count == 2
    assert info.width == 16
    assert info.height == 12
    assert [metric.source_stem for metric in metrics] == ["pair_000001", "pair_000002"]
    assert metrics[0].pair_delta_ms == approx(1.5)


def test_write_stereo_frames_and_realityscan_xmp(tmp_path: Path):
    session = load_stereo_session(_make_stereo_session(tmp_path))
    calibration = load_stereo_calibration(_make_calibration(tmp_path))
    _, selected = read_stereo_session_metrics(session, max_pair_delta_ms=75.0)
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()

    result = write_stereo_variant_frames(
        session,
        selected,
        frames_dir,
        VariantSpec(name="enhanced_brown4", geometry_mode="enhanced"),
        crop_fraction=0.0,
        wb_gain=2.0,
        clahe_clip=2.0,
        sharpen=0.0,
        jpeg_quality=95,
        texture_layers=False,
        calibration=calibration,
        distortion_model="Brown4WithTangential2",
        xmp_pose_prior="exact",
        xmp_calibration_prior="exact",
        translation_scale=0.001,
        include_rig_priors=False,
    )

    assert len(result.image_paths) == 4
    assert len(result.contact_paths) == 2
    right_xmp = frames_dir / "pair_000001_right_t_0000.000.xmp"
    text = right_xmp.read_text(encoding="utf-8")
    assert 'xcr:CalibrationPrior="exact"' in text
    assert 'xcr:CalibrationGroup="2"' in text
    assert 'xcr:DistortionModel="brown4t2"' in text
    assert "xcr:Rig" not in text
    assert "xcr:RigPoseIndex" not in text
    assert "xcr:Position" not in text


def test_stereo_texture_layers_write_color_texture_sidecars(tmp_path: Path):
    session = load_stereo_session(_make_stereo_session(tmp_path))
    _, selected = read_stereo_session_metrics(session, max_pair_delta_ms=75.0)
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()

    result = write_stereo_variant_frames(
        session,
        selected,
        frames_dir,
        VariantSpec(name="flat_luma_kplus", geometry_mode="flat_luma"),
        crop_fraction=0.0,
        wb_gain=2.0,
        clahe_clip=2.0,
        sharpen=0.0,
        jpeg_quality=95,
        texture_layers=True,
        calibration=None,
        distortion_model=DEFAULT_FAST_DISTORTION_MODEL,
        xmp_pose_prior="exact",
        xmp_calibration_prior="exact",
        translation_scale=0.001,
        include_rig_priors=False,
    )

    texture_paths = sorted(frames_dir.glob("*.jpg.texture.jpg"))
    assert len(result.image_paths) == 4
    assert len(texture_paths) == 4
    texture = cv2.imread(str(texture_paths[0]), cv2.IMREAD_COLOR)
    geometry_path = frames_dir / texture_paths[0].name.removesuffix(".texture.jpg")
    geometry = cv2.imread(str(geometry_path), cv2.IMREAD_COLOR)
    assert texture is not None
    assert geometry is not None
    assert not np.array_equal(texture, geometry)


def test_stereo_xmp_mode_does_not_clobber_camera_groups():
    class Args:
        using_stereo_xmp_priors = True

    commands = selected_image_prior_commands(Args())

    assert commands == ["-setFeatureSource 2"]


def test_missing_realityscan_model_is_failure(tmp_path: Path):
    missing_model = tmp_path / "missing.obj"

    assert final_reconstruction_exit_code(0, missing_model) == MISSING_REALITYSCAN_OUTPUT_EXIT_CODE


def _write_solved_xmp(path: Path, position: tuple[float, float, float]) -> None:
    path.write_text(
        f"""<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description xmlns:xcr="http://www.capturingreality.com/ns/xcr/1.1#">
      <xcr:Position>{position[0]} {position[1]} {position[2]}</xcr:Position>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
""",
        encoding="utf-8",
    )


def _write_unsolved_xmp(path: Path) -> None:
    path.write_text(
        """<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description xmlns:xcr="http://www.capturingreality.com/ns/xcr/1.1#">
      <xcr:FocalLength35mm>24</xcr:FocalLength35mm>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
""",
        encoding="utf-8",
    )


def test_metric_scale_from_solved_stereo_xmp_writes_meter_obj(tmp_path: Path):
    calibration = load_stereo_calibration(_make_calibration(tmp_path))
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for index, x_offset in enumerate((0.0, 5.0, 10.0), start=1):
        stem = f"pair_{index:06d}"
        _write_solved_xmp(frames_dir / f"{stem}_left_t_0000.000.xmp", (x_offset, 0.0, 0.0))
        _write_solved_xmp(frames_dir / f"{stem}_right_t_0000.000.xmp", (x_offset + 2.0, 0.0, 0.0))

    model = tmp_path / "model.obj"
    model.write_text(
        "mtllib model.mtl\n"
        "v 1.0 2.0 3.0\n"
        "v -2.0 0.0 4.0 0.4 0.5 0.6\n"
        "vn 0.0 0.0 1.0\n"
        "f 1 2 1\n",
        encoding="utf-8",
    )

    result = scale_model_from_stereo_baseline(
        model,
        frames_dir,
        calibration,
        translation_scale=0.001,
        min_pairs=3,
        report_path=tmp_path / "metric_scale.json",
    )

    assert result.real_baseline_m == approx(0.1)
    assert result.reconstructed_baseline_units == approx(2.0)
    assert result.scale_factor == approx(0.05)
    text = result.metric_model.read_text(encoding="utf-8")
    assert "v 0.05 0.1 0.15" in text
    assert "v -0.1 0 0.2 0.4 0.5 0.6" in text
    assert "vn 0.0 0.0 1.0" in text


def test_metric_scale_uses_numbered_realityscan_solved_xmp_exports(tmp_path: Path):
    calibration = load_stereo_calibration(_make_calibration(tmp_path))
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    positions = (
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (5.0, 0.0, 0.0),
        (7.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (12.0, 0.0, 0.0),
    )
    for index in range(1, 4):
        stem = f"pair_{index:06d}"
        _write_unsolved_xmp(frames_dir / f"{stem}_left_t_0000.000.xmp")
        _write_unsolved_xmp(frames_dir / f"{stem}_right_t_0000.000.xmp")
    for index, position in enumerate(positions):
        _write_solved_xmp(frames_dir / f"{index:05d}.xmp", position)

    model = tmp_path / "model.obj"
    model.write_text("v 1.0 0.0 0.0\n", encoding="utf-8")

    result = scale_model_from_stereo_baseline(
        model,
        frames_dir,
        calibration,
        translation_scale=0.001,
        min_pairs=3,
        report_path=tmp_path / "metric_scale.json",
    )

    assert result.reconstructed_baseline_units == approx(2.0)
    assert result.scale_factor == approx(0.05)
    assert result.pair_count == 3
    assert "v 0.05 0 0" in result.metric_model.read_text(encoding="utf-8")


def test_metric_scale_honors_centimeter_calibration_units(tmp_path: Path):
    calibration = load_stereo_calibration(_make_calibration(tmp_path, units="cm", baseline=12.0))
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for index, x_offset in enumerate((0.0, 5.0, 10.0), start=1):
        stem = f"pair_{index:06d}"
        _write_solved_xmp(frames_dir / f"{stem}_left_t_0000.000.xmp", (x_offset, 0.0, 0.0))
        _write_solved_xmp(frames_dir / f"{stem}_right_t_0000.000.xmp", (x_offset + 2.0, 0.0, 0.0))

    model = tmp_path / "model.obj"
    model.write_text("v 1.0 0.0 0.0\n", encoding="utf-8")

    result = scale_model_from_stereo_baseline(
        model,
        frames_dir,
        calibration,
        translation_scale=0.001,
        min_pairs=3,
        report_path=tmp_path / "metric_scale.json",
    )

    assert calibration.baseline_mm == approx(120.0)
    assert result.real_baseline_m == approx(0.12)
    assert result.scale_factor == approx(0.06)
    assert "v 0.06 0 0" in result.metric_model.read_text(encoding="utf-8")


def test_final_command_exports_solved_xmp_for_metric_scaling(tmp_path: Path):
    class Args:
        detector_sensitivity = "Ultra"
        distortion_model = "Brown4WithTangential2"
        images_overlap = "Low"
        max_features_per_image = 80000
        max_features_per_mpx = 20000
        metric_scale_active = True
        model_quality = "preview"
        normal_downscale = 2
        preselector_features = 20000
        recon_region_scale_xy = 1.25
        recon_region_scale_z = 1.35
        simplify_triangles = 0
        texture_count = 1
        texture_layers = False
        texture_resolution = 512
        try_merge_components = False
        using_stereo_xmp_priors = False

    paths = prepare_output_paths(tmp_path / "session", tmp_path / "out", overwrite=False)

    write_realityscan_command_file(paths, Args(), frames_dir=tmp_path / "frames")

    text = paths.rscmd.read_text(encoding="utf-8")
    assert "-exportXMPForSelectedComponent" in text
    assert text.index("-exportModel") < text.index("-exportXMPForSelectedComponent")


def test_final_command_uses_texture_layer_for_coloring_and_texturing(tmp_path: Path):
    class Args:
        detector_sensitivity = "Ultra"
        distortion_model = DEFAULT_FAST_DISTORTION_MODEL
        images_overlap = "Low"
        max_features_per_image = 80000
        max_features_per_mpx = 20000
        metric_scale_active = False
        model_quality = "preview"
        normal_downscale = 2
        preselector_features = 20000
        recon_region_scale_xy = 1.25
        recon_region_scale_z = 1.35
        simplify_triangles = 0
        texture_count = 1
        texture_layers = True
        texture_resolution = 512
        try_merge_components = False
        using_stereo_xmp_priors = False

    paths = prepare_output_paths(tmp_path / "session", tmp_path / "out", overwrite=False)

    write_realityscan_command_file(paths, Args(), frames_dir=tmp_path / "frames")

    text = paths.rscmd.read_text(encoding="utf-8")
    assert '-set "ImageLayerForColoring=texture01"' in text
    assert '-set "ImageLayerForTexturing=texture01"' in text
    assert text.index("-addFolder") < text.index("ImageLayerForTexturing=texture01")
    assert text.index("ImageLayerForTexturing=texture01") < text.index("-selectAllImages")


def test_final_command_skips_realityscan_clean_model_by_default(tmp_path: Path):
    class Args:
        detector_sensitivity = "Ultra"
        distortion_model = "Brown4WithTangential2"
        images_overlap = "Low"
        max_features_per_image = 80000
        max_features_per_mpx = 20000
        metric_scale_active = False
        model_quality = "preview"
        normal_downscale = 2
        preselector_features = 20000
        recon_region_scale_xy = 1.25
        recon_region_scale_z = 1.35
        simplify_triangles = 0
        texture_count = 1
        texture_layers = False
        texture_resolution = 512
        try_merge_components = False
        using_stereo_xmp_priors = False

    paths = prepare_output_paths(tmp_path / "session", tmp_path / "out", overwrite=False)

    write_realityscan_command_file(paths, Args(), frames_dir=tmp_path / "frames")

    text = paths.rscmd.read_text(encoding="utf-8")
    assert "-cleanModel" not in text
    assert '-exportModel "Model 1"' in text


def test_final_command_can_still_use_realityscan_clean_model(tmp_path: Path):
    class Args:
        clean_model = True
        detector_sensitivity = "Ultra"
        distortion_model = "Brown4WithTangential2"
        images_overlap = "Low"
        max_features_per_image = 80000
        max_features_per_mpx = 20000
        metric_scale_active = False
        model_quality = "preview"
        normal_downscale = 2
        preselector_features = 20000
        recon_region_scale_xy = 1.25
        recon_region_scale_z = 1.35
        simplify_triangles = 0
        texture_count = 1
        texture_layers = False
        texture_resolution = 512
        try_merge_components = False
        using_stereo_xmp_priors = False

    paths = prepare_output_paths(tmp_path / "session", tmp_path / "out", overwrite=False)

    write_realityscan_command_file(paths, Args(), frames_dir=tmp_path / "frames")

    text = paths.rscmd.read_text(encoding="utf-8")
    assert "-cleanModel" in text
    assert '-exportModel "Model 2"' in text


def test_large_face_filter_removes_broad_infill_triangles(tmp_path: Path):
    model = tmp_path / "model.obj"
    model.write_text(
        "mtllib model.mtl\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "v 100 0 0\n"
        "v 0 100 0\n"
        "usemtl surface\n"
        "f 1 2 3\n"
        "f 1 2 3\n"
        "f 1 2 3\n"
        "f 1 2 3\n"
        "f 1 2 3\n"
        "f 1 4 5\n",
        encoding="utf-8",
    )

    result = filter_obj_large_faces(
        model,
        area_ratio=20.0,
        min_faces=1,
        report_path=tmp_path / "filter.json",
    )

    text = model.read_text(encoding="utf-8")
    assert result.face_count == 6
    assert result.removed_face_count == 1
    assert text.count("f 1 2 3") == 5
    assert "f 1 4 5" not in text


def test_high_detail_preset_raises_alignment_and_reconstruction_budget():
    parser = build_arg_parser()
    args = parser.parse_args(["input.mp4", "--reconstruction-preset", "high-detail"])

    applied = apply_reconstruction_preset(args, parser)

    assert applied["model_quality"] == "high"
    assert args.normal_downscale == 1
    assert args.max_frames == 720
    assert args.max_features_per_image == 160000
    assert args.simplify_triangles == 4000000
    assert args.texture_resolution == 8192


def test_detail_preset_keeps_explicit_cli_overrides():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "input.mp4",
            "--reconstruction-preset",
            "max-detail",
            "--simplify-triangles",
            "2500000",
            "--max-frames",
            "300",
        ]
    )

    applied = apply_reconstruction_preset(args, parser)

    assert "simplify_triangles" not in applied
    assert "max_frames" not in applied
    assert args.simplify_triangles == 2500000
    assert args.max_frames == 300
    assert args.model_quality == "high"
    assert args.normal_downscale == 1


def test_raw_geometry_mode_leaves_frame_unenhanced():
    frame = np.arange(4 * 5 * 3, dtype=np.uint8).reshape((4, 5, 3))

    raw = make_geometry_frame(frame, "raw", wb_gain=2.0, clahe_clip=2.0, sharpen=0.5)
    enhanced = make_geometry_frame(frame, "enhanced", wb_gain=2.0, clahe_clip=2.0, sharpen=0.5)

    assert np.array_equal(raw, frame)
    assert raw is not frame
    assert not np.array_equal(enhanced, frame)


def _metric(frame_index: int, timestamp_s: float, quality: float, fingerprint_level: int, caustic: float = 0.0) -> FrameMetric:
    fingerprint = np.full((8, 8), fingerprint_level, dtype=np.uint8)
    return FrameMetric(
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        sharpness=1.0,
        contrast=1.0,
        brightness=128.0,
        feature_count=100,
        exposure_score=1.0,
        quality=quality,
        caustic_score=caustic,
        fingerprint=fingerprint,
    )


def test_connectivity_bridge_selection_adds_mid_gap_frames():
    metrics = [
        _metric(0, 0.0, 1.0, 0),
        _metric(1, 0.33, 0.52, 28),
        _metric(2, 0.66, 0.51, 56),
        _metric(3, 1.0, 1.0, 96),
    ]

    selected = select_frames(
        metrics,
        target_fps=1.0,
        max_frames=4,
        min_frames=2,
        quality_quantile=0.0,
        min_motion=0.0,
        max_still_gap_s=2.0,
        connectivity_bridge_selection=True,
        bridge_max_gap_s=0.4,
        bridge_max_delta=8.0,
        bridge_quality_floor=0.2,
        bridge_max_extra_fraction=1.0,
    )

    assert [metric.frame_index for metric in selected] == [0, 1, 2, 3]


def test_connectivity_bridge_selection_can_be_disabled():
    metrics = [
        _metric(0, 0.0, 1.0, 0),
        _metric(1, 0.33, 0.52, 28),
        _metric(2, 0.66, 0.51, 56),
        _metric(3, 1.0, 1.0, 96),
    ]

    selected = select_frames(
        metrics,
        target_fps=1.0,
        max_frames=4,
        min_frames=2,
        quality_quantile=0.0,
        min_motion=0.0,
        max_still_gap_s=2.0,
        connectivity_bridge_selection=False,
    )

    assert [metric.frame_index for metric in selected] == [0, 3]


def test_caustic_luma_suppresses_bright_low_saturation_ripples():
    frame = np.full((80, 120, 3), 75, dtype=np.uint8)
    frame[:, 20:25] = 235
    frame[:, 65:70] = 245

    raw_score = estimate_caustic_score(frame)
    stable = make_geometry_frame(frame, "caustic_luma", wb_gain=2.0, clahe_clip=2.0, sharpen=0.0)
    flat = make_geometry_frame(frame, "flat_luma", wb_gain=2.0, clahe_clip=2.0, sharpen=0.0)

    assert raw_score > 0.15
    assert stable[:, 20:25].mean() < flat[:, 20:25].mean()


def test_fast_default_uses_flat_luma_kplus_without_tournament():
    parser = build_arg_parser()
    args = parser.parse_args(["input.mp4"])

    variants = build_variant_specs(args)

    assert args.alignment_tournament == "off"
    assert args.base_geometry_mode == DEFAULT_FAST_GEOMETRY_MODE
    assert args.distortion_model == DEFAULT_FAST_DISTORTION_MODEL
    assert args.texture_layers is True
    assert args.min_good_component_ratio == approx(DEFAULT_MIN_GOOD_COMPONENT_RATIO)
    assert len(variants) == 1
    assert variants[0].name == "flat_luma_kplus"
    assert variants[0].geometry_mode == "flat_luma"
    assert variants[0].distortion_model == DEFAULT_FAST_DISTORTION_MODEL


def test_legacy_enhanced_default_switch_restores_old_fast_variant():
    parser = build_arg_parser()
    args = parser.parse_args(["input.mp4", "--legacy-enhanced-default"])
    apply_legacy_enhanced_default(args)

    variants = build_variant_specs(args)

    assert args.base_geometry_mode == LEGACY_GEOMETRY_MODE
    assert args.distortion_model == LEGACY_DISTORTION_MODEL
    assert len(variants) == 1
    assert variants[0].name == "enhanced_brown4"
    assert variants[0].geometry_mode == "enhanced"
    assert variants[0].distortion_model == LEGACY_DISTORTION_MODEL


def test_raw_base_geometry_mode_builds_raw_variant():
    parser = build_arg_parser()
    args = parser.parse_args(["input.mp4", "--base-geometry-mode", "raw"])

    variants = build_variant_specs(args)

    assert len(variants) == 1
    assert variants[0].name == "raw_kplus"
    assert variants[0].geometry_mode == "raw"


def test_caustic_luma_base_geometry_mode_builds_variant():
    parser = build_arg_parser()
    args = parser.parse_args(["input.mp4", "--base-geometry-mode", "caustic_luma"])

    variants = build_variant_specs(args)

    assert len(variants) == 1
    assert variants[0].name == "caustic_luma_kplus"
    assert variants[0].geometry_mode == "caustic_luma"


def _component_report(*components: tuple[str, int, int, int, float]) -> str:
    sections = []
    for name, registered, points, projections, reprojection in components:
        sections.append(
            f"""
            <p class="itemTitle">Component: {name}</p>
            <table class="propertiesTable">
                <tr><th>Count of registered images</th><td>{registered} /  20 </td></tr>
                <tr><th>Points' count</th><td>{points}</td></tr>
            </table>
            <p class="componentSubTitle">Alignment report<p>
            <table class="propertiesTable">
                <tr><th>Total projections</th><td>{projections}</td></tr>
                <tr><th>Average track length</th><td>2.5</td></tr>
                <tr><th>Mean reprojection error [pixels]</th><td>{reprojection}</td></tr>
            </table>
            """
        )
    return "\n".join(sections)


def test_parse_alignment_report_keeps_ranked_component_details(tmp_path: Path):
    report = tmp_path / "overview.html"
    report.write_text(
        _component_report(
            ("Component 0", 2, 900, 1800, 0.51),
            ("Component 7", 9, 4200, 9800, 0.76),
            ("Component 4", 5, 2500, 5400, 0.62),
        ),
        encoding="utf-8",
    )

    result = parse_alignment_report(
        report,
        name="flat_luma_kplus",
        selected_image_count=20,
        project=tmp_path / "aligned.rsproj",
    )

    assert result.component_count == 3
    assert result.largest_component_images == 9
    assert result.largest_component_ratio == approx(0.45)
    assert [component.name for component in pipeline.ranked_components(result.components)] == [
        "Component 7",
        "Component 4",
        "Component 0",
    ]
    assert result.components[1].points_count == 4200
    assert result.components[1].mean_reprojection_error_px == approx(0.76)


def test_write_alignment_results_includes_component_table(tmp_path: Path):
    paths = prepare_output_paths(tmp_path / "session", tmp_path / "out", overwrite=False)
    report = paths.reports / "overview.html"
    report.write_text(
        _component_report(
            ("Component 0", 3, 1200, 2400, 0.4),
            ("Component 2", 8, 3800, 9000, 0.7),
        ),
        encoding="utf-8",
    )
    result = parse_alignment_report(
        report,
        name="flat_luma_kplus",
        selected_image_count=20,
        project=paths.project,
    )

    write_alignment_results(paths, [result])

    component_csv = paths.reports / "alignment_components.csv"
    component_json = paths.reports / "alignment_components.json"
    assert component_csv.exists()
    assert component_json.exists()
    csv_text = component_csv.read_text(encoding="utf-8")
    assert "flat_luma_kplus,1,Component 2,8,20,0.400000" in csv_text
    assert "flat_luma_kplus,2,Component 0,3,20,0.150000" in csv_text
    data = json.loads(component_json.read_text(encoding="utf-8"))
    assert data[0]["component"] == "Component 2"
    assert data[0]["registered_images"] == 8


def test_alignment_only_runs_single_fast_variant_when_tournament_is_off(tmp_path: Path, monkeypatch):
    parser = build_arg_parser()
    args = parser.parse_args(["input.mp4", "--alignment-only"])
    spec = build_variant_specs(args)[0]
    paths = prepare_output_paths(tmp_path / "session", tmp_path / "out", overwrite=False)
    variant_paths = {spec.name: make_variant_paths(paths, spec)}
    calls: list[str] = []

    def fake_run_realityscan_rscmd(**kwargs):
        calls.append(str(kwargs["label"]))
        variant_paths[spec.name].report.write_text(
            _component_report(("Component 1", 6, 3000, 7000, 0.5)),
            encoding="utf-8",
        )
        variant_paths[spec.name].project.write_text("project\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(pipeline, "run_realityscan_rscmd", fake_run_realityscan_rscmd)

    results = run_alignment_tournament(
        paths,
        [spec],
        variant_paths,
        selected_count=20,
        realityscan_exe=tmp_path / "RealityScan.exe",
        args=args,
    )

    assert calls == ["alignment variant flat_luma_kplus"]
    assert len(results) == 1
    assert results[0].largest_component_images == 6
