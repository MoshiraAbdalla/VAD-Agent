import argparse
import importlib.util
import json
import pathlib
import sys

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scripts.plot_raw_auc_qualitative_curves as raw_curves


VERA_ROOT = pathlib.Path(r"D:\Moshira Research\VERA-main-cvpr2025\MDF_VAD")
OLD_RENDERER = VERA_ROOT / "render_qualitative_vad_agent_lavad_figures.py"
FRAME_ROOTS = {
    "ucf": VERA_ROOT / "data" / "ucf" / "frames",
    "xd": VERA_ROOT / "data" / "xd" / "frames",
}
AGENT_ROOTS = {
    "ucf": ROOT / "results" / "predictions" / "ucf_crime_mif_stride150",
    "xd": ROOT / "results" / "predictions" / "xd_violence_mif_stride64_2",
}
OUT_ROOTS = {
    "ucf": ROOT / "results" / "videos" / "ucf_crime",
    "xd": ROOT / "results" / "videos" / "xd_stride64",
}


def load_old_renderer():
    spec = importlib.util.spec_from_file_location("old_qual_renderer", OLD_RENDERER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_json(path):
    with pathlib.Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def gt_ranges(sample):
    labels = sample.get("temporal_label", [])
    ranges = []
    for i in range(0, len(labels), 2):
        start = int(labels[i])
        if start < 0:
            continue
        end = int(min(labels[i + 1], sample["length"]))
        if end > start:
            ranges.append((start, end))
    return ranges


def frame_in_ranges(frame_idx, ranges):
    return any(int(start) <= int(frame_idx) < int(end) for start, end in ranges)


def paste_fill(canvas, image, box):
    x0, y0, x1, y1 = box
    image = image.resize((x1 - x0, y1 - y0), Image.LANCZOS)
    canvas.paste(image, (x0, y0))


def fit_image(image, box_w, box_h, bg=(245, 245, 245)):
    out = Image.new("RGB", (box_w, box_h), bg)
    img = image.copy()
    scale = min(box_w / max(1, img.width), box_h / max(1, img.height))
    new_w = max(1, int(round(img.width * scale)))
    new_h = max(1, int(round(img.height * scale)))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    x = (box_w - img.width) // 2
    y = (box_h - img.height) // 2
    out.paste(img, (x, y))
    return out, (x, y, img.width, img.height)


def build_plot_base(video, ranges, length, title, width=1920, height=240):
    fig, ax = plt.subplots(1, 1, figsize=(width / 100, height / 100), dpi=100)
    for start, end in ranges:
        ax.axvspan(start, min(end, length - 1), color="#e85b64", alpha=0.42, linewidth=0)
    ax.set_xlim(0, max(1, length - 1))
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("Frame number")
    ax.set_title(f"{video}: {title}")
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.canvas.draw()
    box = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    bbox = (
        int(round(box.x0 * fig.dpi)),
        int(round(height - box.y1 * fig.dpi)),
        int(round(box.x1 * fig.dpi)),
        int(round(height - box.y0 * fig.dpi)),
    )
    canvas = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return Image.fromarray(canvas, "RGB"), bbox


def build_two_plot_base(video, ranges, length, width=1920, height=240):
    fig, axes = plt.subplots(2, 1, figsize=(width / 100, height / 100), dpi=100, sharex=True)
    titles = [f"{video}: Direct segment score", "Final temporal score"]
    for ax, title in zip(axes, titles):
        for start, end in ranges:
            ax.axvspan(start, min(end, length - 1), color="#e85b64", alpha=0.42, linewidth=0)
        ax.set_xlim(0, max(1, length - 1))
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.grid(True, alpha=0.22)
    axes[1].set_xlabel("Frame number")
    fig.tight_layout()
    fig.canvas.draw()
    bboxes = []
    for ax in axes:
        box = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        bboxes.append(
            (
                int(round(box.x0 * fig.dpi)),
                int(round(height - box.y1 * fig.dpi)),
                int(round(box.x1 * fig.dpi)),
                int(round(height - box.y0 * fig.dpi)),
            )
        )
    canvas = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return Image.fromarray(canvas, "RGB"), bboxes


def draw_score_curve(draw, bbox, scores, frame_idx, length, label=None, color=(68, 119, 170), cursor=True):
    x0, y0, x1, y1 = bbox
    upto = max(0, min(int(frame_idx), min(int(length) - 1, len(scores) - 1)))
    if upto > 0:
        xs = np.arange(upto + 1, dtype=np.float32)
        ys = np.asarray(scores[: upto + 1], dtype=np.float32)
        px = x0 + (xs / max(1, int(length) - 1)) * (x1 - x0)
        py = y1 - (np.clip(ys, 0.0, 1.05) / 1.05) * (y1 - y0)
        points = [(int(round(a)), int(round(b))) for a, b in zip(px, py)]
        if len(points) > 1:
            draw.line(points, fill=color, width=4, joint="curve")
    if cursor:
        cursor_x = x0 + int(round((upto / max(1, int(length) - 1)) * (x1 - x0)))
        draw.line((cursor_x, y0, cursor_x, y1), fill=(214, 39, 40), width=3)
    if label:
        legend_x = max(x0 + 8, x1 - 430)
        legend_y = y1 - 32
        draw.rectangle((legend_x - 8, legend_y - 8, x1 - 8, y1 - 6), fill=(255, 255, 255), outline=(210, 210, 210))
        draw.line((legend_x, legend_y + 5, legend_x + 46, legend_y + 5), fill=color, width=4)
        draw.text((legend_x + 56, legend_y - 4), label, fill=(0, 0, 0))


def plot_with_revealed_score(base_img, bbox, scores, frame_idx, length, label):
    img = base_img.copy()
    draw_score_curve(ImageDraw.Draw(img), bbox, scores, frame_idx, length, label=label)
    return img


def plot_with_direct_and_temporal(base_img, bboxes, direct, final, frame_idx, length):
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    draw_score_curve(draw, bboxes[0], direct, length - 1, length, label="Direct segment score", cursor=False)
    draw_score_curve(
        draw,
        bboxes[1],
        final,
        frame_idx,
        length,
        label="Final temporal score",
        color=(34, 139, 92),
    )
    return img


def load_frame(dataset, video, frame_idx):
    path = FRAME_ROOTS[dataset] / video / f"{max(0, int(frame_idx)):06d}.jpg"
    if not path.exists():
        frames = sorted((FRAME_ROOTS[dataset] / video).glob("*.jpg"))
        if not frames:
            raise FileNotFoundError(f"No frames found for {dataset}:{video}")
        path = frames[min(max(0, int(frame_idx)), len(frames) - 1)]
    return Image.open(path).convert("RGB")


def projected_segment_for_frame(payload, frame_idx):
    segments = [s for s in payload["segments"] if "agent_outputs" in s]
    if not segments:
        raise ValueError(f"No usable agent segments in {payload.get('video')}")
    stride = int(payload.get("segment_stride", 0) or 0)
    if stride <= 0 and len(segments) > 1:
        starts = np.asarray([int(s.get("start", 0)) for s in segments], dtype=np.int32)
        diffs = np.diff(starts)
        stride = int(np.median(diffs)) if len(diffs) else 0
    if stride <= 0:
        stride = int(payload.get("segment_len", 1) or 1)
    idx = int(max(0, int(frame_idx)) // max(1, stride))
    idx = min(idx, len(segments) - 1)
    return segments[idx]


def render_video(args):
    dataset = args.dataset
    video = args.video
    render = load_old_renderer()
    payload = load_json(AGENT_ROOTS[dataset] / f"{video}.json")
    curves = raw_curves.compute_curves(
        dataset,
        video,
        score_folder=AGENT_ROOTS[dataset] if dataset == "xd" else None,
    )
    direct = curves["direct"]
    final = curves["final"]
    sample = curves["sample"]
    ranges = gt_ranges(sample)
    length = int(sample["length"])

    out_dir = OUT_ROOTS[dataset]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = pathlib.Path(args.out_video)
    if not out_path.is_absolute():
        out_path = out_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    enc_w = int(args.width)
    enc_h = int(args.height)
    plot_h = int(args.plot_height)
    top_h = enc_h - plot_h
    left_w = int(enc_w * float(args.video_width_ratio))
    right_w = enc_w - left_w
    direct_plot_base, direct_plot_bbox = build_plot_base(
        video,
        ranges,
        length,
        "direct segment score",
        width=enc_w,
        height=plot_h,
    )
    final_two_plot_base, final_two_plot_bboxes = build_two_plot_base(video, ranges, length, width=enc_w, height=plot_h)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(args.fps), (enc_w, enc_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {out_path}")

    frame_step = max(1, int(args.frame_step))
    for fid in range(0, length, frame_step):
        seg = projected_segment_for_frame(payload, fid)
        frame = load_frame(dataset, video, fid)
        frame_panel, (fx, fy, fw, fh) = fit_image(frame, left_w, top_h)
        text_panel = render.draw_agent_box(seg["agent_outputs"], (right_w, top_h))
        plot = plot_with_revealed_score(direct_plot_base, direct_plot_bbox, direct, fid, length, "Direct segment score")

        canvas = Image.new("RGB", (enc_w, enc_h), (248, 248, 248))
        canvas.paste(frame_panel, (0, 0))
        border = (225, 35, 35) if frame_in_ranges(fid, ranges) else (0, 35, 255)
        d = ImageDraw.Draw(canvas)
        d.rectangle((fx, fy, fx + fw, fy + fh), outline=border, width=5)
        d.text((18, 18), "Frame", font=render.FONT_TITLE, fill=(0, 0, 0))
        d.text((18, 68), f"{fid}", font=render.FONT_TITLE, fill=(0, 0, 0))
        canvas.paste(text_panel, (left_w, 0))
        paste_fill(canvas, plot, (0, top_h, enc_w, enc_h))
        writer.write(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))

    final_frame_idx = max(0, length - 1)
    final_seg = projected_segment_for_frame(payload, final_frame_idx)
    final_frame = load_frame(dataset, video, final_frame_idx)
    final_frame_panel, (fx, fy, fw, fh) = fit_image(final_frame, left_w, top_h)
    final_text_panel = render.draw_agent_box(final_seg["agent_outputs"], (right_w, top_h))
    reveal_frames = max(1, int(round(float(args.fps) * float(args.temporal_reveal_seconds))))
    hold_frames = max(0, int(round(float(args.fps) * float(args.temporal_hold_seconds))))
    for i in range(reveal_frames + hold_frames):
        if i < reveal_frames:
            fid = int(round((i / max(1, reveal_frames - 1)) * (length - 1)))
        else:
            fid = final_frame_idx
        plot = plot_with_direct_and_temporal(final_two_plot_base, final_two_plot_bboxes, direct, final, fid, length)
        canvas = Image.new("RGB", (enc_w, enc_h), (248, 248, 248))
        canvas.paste(final_frame_panel, (0, 0))
        border = (225, 35, 35) if frame_in_ranges(final_frame_idx, ranges) else (0, 35, 255)
        d = ImageDraw.Draw(canvas)
        d.rectangle((fx, fy, fx + fw, fy + fh), outline=border, width=5)
        d.text((18, 18), "Frame", font=render.FONT_TITLE, fill=(0, 0, 0))
        d.text((18, 68), f"{final_frame_idx}", font=render.FONT_TITLE, fill=(0, 0, 0))
        canvas.paste(final_text_panel, (left_w, 0))
        paste_fill(canvas, plot, (0, top_h, enc_w, enc_h))
        writer.write(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"saved_video={out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ucf", "xd"], default="ucf")
    parser.add_argument("--video", default="Explosion033_x264")
    parser.add_argument("--out-video", default="vad_agent_qualitative.mp4")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--frame-step", type=int, default=3)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--plot-height", type=int, default=300)
    parser.add_argument("--video-width-ratio", type=float, default=0.72)
    parser.add_argument("--temporal-reveal-seconds", type=float, default=4.0)
    parser.add_argument("--temporal-hold-seconds", type=float, default=2.0)
    args = parser.parse_args()
    render_video(args)


if __name__ == "__main__":
    main()



