"""Render qualitative examples with complete saved agent outputs."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CANVAS_WIDTH = 2400
PRINT_WIDTH = 7.1
EXPORT_DPI = CANVAS_WIDTH / PRINT_WIDTH
BODY_SIZE = 45
FIELDS = [("Summarization Agent Output:", "summarization_agent", (230, 35, 35)),
          ("Context Agent Output:", "context_agent", (105, 185, 95)),
          ("Reasoning Agent Output:", "reasoning_agent", (20, 70, 185))]
SPECS = [
    {"id": "qualitative_ucf_abnormal", "dataset": "ucf", "video": "Explosion033_x264", "kind": "abnormal", "title": "UCF-Crime abnormal: Explosion033_x264", "frames": [300, 2100]},
    {"id": "qualitative_xd_abnormal", "dataset": "xd", "video": "Mission.Impossible.Fallout.2018__#00-31-21_00-32-50_label_B1-0-0", "kind": "abnormal", "title": "XD-Violence abnormal: Mission.Impossible.Fallout.2018", "frames": [100, 1200]},
    {"id": "qualitative_ucf_normal", "dataset": "ucf", "video": "Normal_Videos_050_x264", "kind": "normal", "title": "UCF-Crime normal: Normal_Videos_050_x264", "frames": [2250]},
    {"id": "qualitative_xd_normal", "dataset": "xd", "video": "The.Bourne.Ultimatum.2007__#0-55-55_0-56-10_label_A", "kind": "normal", "title": "XD-Violence normal: The.Bourne.Ultimatum.2007", "frames": [75]},
]

def font(size, bold=False):
    names = ["arialbd.ttf" if bold else "arial.ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()

def wrap_lines(text, width_px, font_obj):
    lines=[]
    for paragraph in str(text).splitlines():
        if not paragraph.strip():
            lines.append("");continue
        current=""
        for word in paragraph.split():
            candidate=(current+" "+word).strip()
            if current and font_obj.getlength(candidate)>width_px:
                lines.append(current);current=word
            else:
                current=candidate
        if current:lines.append(current)
    return lines


def draw_wrapped(draw, xy, text, fill, width_px, font_obj, line_gap=5):
    x, y = xy
    for line in wrap_lines(text, width_px, font_obj):
        if line:
            draw.text((x, y), line, font=font_obj, fill=fill)
        y += font_obj.size + line_gap
    return y

def clean_agent_text(text):
    text = text.replace("Context Summary:", "").replace("Summary:", "").strip()
    text = text.replace("Answers to Prompt Questions:", "").strip()
    return text

def split_reasoning_output(text):
    text = clean_agent_text(text)
    match = re.search(r"\bOutput:\s*([01])\s*[\]\)]?\s*$", text, flags=re.IGNORECASE)
    if not match:
        return text, "", (0, 0, 0)
    value = match.group(1)
    body = text[: match.start()].rstrip(" \n[]")
    color = (0, 55, 255) if value == "0" else (230, 35, 35)
    return body, f"Output: {value}", color

def text_width(draw, text, font_obj):
    if not text:
        return 0
    box = draw.textbbox((0, 0), text, font=font_obj)
    return box[2] - box[0]

def reasoning_lines(text, width_px, font_obj):
    body, output, output_color = split_reasoning_output(text)
    lines = wrap_lines(body, width_px, font_obj) or [""]
    needed = font_obj.getlength(lines[-1] + "  ") + font(font_obj.size, True).getlength(output)
    extra_output_line = bool(output and needed > width_px)
    return lines, output, output_color, extra_output_line


def reasoning_line_count(text, width_px, font_obj):
    lines, output, color, extra = reasoning_lines(text, width_px, font_obj)
    return len(lines) + int(extra)


def draw_wrapped_reasoning(draw, xy, text, width_px, font_obj, line_gap=2):
    x, y = xy
    lines, output, output_color, extra = reasoning_lines(text, width_px, font_obj)
    output_font = font(font_obj.size, True)
    for line in lines[:-1]:
        if line:
            draw.text((x, y), line, font=font_obj, fill=(0, 0, 0))
        y += font_obj.size + line_gap
    last = lines[-1]
    if last:
        draw.text((x, y), last, font=font_obj, fill=(0, 0, 0))
    if output:
        if extra:
            y += font_obj.size + line_gap
            out_x = x
        else:
            out_x = x + text_width(draw, last + "  ", font_obj)
        draw.text((out_x, y), output, font=output_font, fill=output_color)
    return y + font_obj.size + line_gap

FONT_TITLE = font(49, True)
FONT_SMALL = font(45)


def draw_agent_box(payload, width):
    x, line_gap = 10, 2
    body_font = font(BODY_SIZE)
    needed = 8
    for label, key, _ in FIELDS:
        text = payload[key]
        count = reasoning_line_count(text, width - 2*x, body_font) if key == "reasoning_agent" else len(wrap_lines(clean_agent_text(text), width - 2*x, body_font))
        needed += BODY_SIZE + line_gap + count * (BODY_SIZE + line_gap) + 5
    image = Image.new("RGB", (width, needed + 10), "white")
    draw = ImageDraw.Draw(image)
    y = 8
    for label, key, color in FIELDS:
        draw.text((x, y), label, font=body_font, fill=color)
        y += BODY_SIZE + line_gap
        if key == "reasoning_agent":
            y = draw_wrapped_reasoning(draw, (x, y), payload[key], width - 2*x, body_font, line_gap)
        else:
            y = draw_wrapped(draw, (x, y), clean_agent_text(payload[key]), (0, 0, 0), width - 2*x, body_font, line_gap)
        y += 5
    assert y <= image.height
    return image


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_case(spec, root, frame_root, raw_curves):
    dataset, video = spec["dataset"], spec["video"]
    score_root = root / ("results/predictions/xd_violence_mif_stride64_2" if dataset == "xd"
                         else raw_curves.DATASETS[dataset]["score_folder"])
    source = score_root / (video + ".json")
    payload = load_json(source)
    segments = sorted(payload["segments"], key=lambda row: int(row["start"]))
    curves = raw_curves.compute_curves(dataset, video, score_folder=score_root)
    if curves["align_mode"] != "exact" or curves["n_segments"] != len(segments):
        raise ValueError(f"Feature and saved-segment alignment differs for {video}.")
    gt = raw_curves.gt_ranges(curves["sample"])
    snapshots = []
    for frame in spec["frames"]:
        segment = next(row for row in reversed(segments) if int(row["start"]) <= frame)
        if not int(segment["start"]) <= frame < int(segment["end"]):
            raise ValueError(f"Frame {frame} is outside its saved segment.")
        outputs = segment["agent_outputs"]
        if not all(outputs.get(key, "").strip() for _, key, _ in FIELDS):
            raise ValueError(f"Full agent outputs are missing for {video}:{segment['start']}.")
        values = re.findall(r"\bOutput\s*:\s*(?:\[\s*)?([01])\b", outputs["reasoning_agent"], re.I)
        if not values or int(values[0]) != int(segment["label"]):
            raise ValueError(f"Reasoning output and decision differ for {video}:{segment['start']}.")
        image_path = frame_root / video / f"{frame:06d}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        snapshots.append({"frame": frame, "start": int(segment["start"]), "end": int(segment["end"]),
                          "label": int(segment["label"]), "frame_path": image_path,
                          "frame_sha256": sha256(image_path),
                          "agent_outputs": {key: outputs[key] for _, key, _ in FIELDS}})
    features = root / raw_curves.DATASETS[dataset]["vision_folder"] / (video + ".npy")
    return {"spec": spec, "snapshots": snapshots, "curves": curves, "ground_truth_intervals": gt,
            "score_source": str(source.relative_to(root)), "score_sha256": sha256(source),
            "agent_source": str(source.relative_to(root)), "agent_sha256": sha256(source),
            "feature_source": str(features.relative_to(root)), "feature_sha256": sha256(features)}


def draw_markers(ax, frames, gt):
    for frame in frames:
        color = "#e6232e" if any(start <= frame < end for start, end in gt) else "#0035ff"
        ax.annotate("", xy=(frame, 0.02), xytext=(frame, 1.06),
                    arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 1.7},
                    annotation_clip=False)


def plot_comparison(case, width_px, out_path, height_inches=2.75):
    fig, axes = plt.subplots(2, 1, figsize=(width_px / EXPORT_DPI, height_inches), dpi=EXPORT_DPI, sharex=True)
    for ax, key, label in zip(axes, ["direct", "final"], ["Direct segment score", "Final refined score"]):
        values = case["curves"][key]
        zero = float(np.max(np.abs(values))) < 1e-8
        ax.plot(np.arange(len(values)), values, color="#1f5f91", linewidth=1.9 if zero else 1.4,
                label=label, zorder=8, solid_capstyle="round", clip_on=False)
        if zero:
            ax.axhline(0, color="#1f5f91", linewidth=1.9, zorder=9, clip_on=False)
        for start, end in case["ground_truth_intervals"]:
            ax.axvspan(start, min(end, len(values)-1), color="#e85b64", alpha=0.42, zorder=1)
        draw_markers(ax, case["spec"]["frames"], case["ground_truth_intervals"])
        ax.set_xlim(0, len(values)-1)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xticks(np.linspace(0, len(values)-1, 6, dtype=int))
        ax.set_ylabel("Score", fontsize=9.7, fontweight="bold")
        ax.grid(True, alpha=0.22, zorder=0)
        ax.tick_params(axis="both", width=0.7, labelsize=9.5, pad=3.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
            spine.set_zorder(2)
        ax.legend(loc="lower right",frameon=True,prop={"weight":"bold","size":9.5},handlelength=1.3,borderpad=0.2)
    axes[1].set_xlabel("Frame number",fontsize=9.7,fontweight="bold",labelpad=4)
    fig.subplots_adjust(left=0.085,right=0.965,bottom=0.20,top=0.97,hspace=0.16)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    from matplotlib.text import Text
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        if artist.get_fontsize() < 9.5:
            raise ValueError("Plot text is below the print-size minimum.")
        box = artist.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1:
            raise ValueError(f"Plot text falls outside the canvas: {artist.get_text()}")
    fig.savefig(out_path, dpi=EXPORT_DPI)
    plt.close(fig)
    with Image.open(out_path) as source:
        result = source.convert("RGB")
        if result.width != width_px:
            result = result.resize((width_px, round(result.height * width_px / result.width)), Image.Resampling.LANCZOS)
        return result


MARGIN = 40
FRAME_WIDTH = 660
IMAGE_HEIGHT = 371


def draw_frame(canvas, snapshot, x, y, width, gt):
    if width != FRAME_WIDTH:
        raise ValueError("All frame panels must use the same width.")
    with Image.open(snapshot["frame_path"]) as source:
        image = source.convert("RGB")
        black_rows = (np.asarray(image).max(axis=2) < 16).mean(axis=1) >= 0.98
        top, bottom = 0, image.height
        while top < bottom and black_rows[top]:
            top += 1
        while bottom > top and black_rows[bottom-1]:
            bottom -= 1
        if top < bottom:
            image = image.crop((0, top, image.width, bottom))
        image = image.resize((width, IMAGE_HEIGHT), Image.Resampling.LANCZOS)
    canvas.paste(image, (x, y))
    color = (230, 35, 35) if any(start <= snapshot["frame"] < end for start, end in gt) else (0, 35, 255)
    ImageDraw.Draw(canvas).rectangle((x, y, x+width-1, y+IMAGE_HEIGHT-1), outline=color, width=5)


def case_metadata(case):
    return {"id": case["spec"]["id"], "dataset": case["spec"]["dataset"], "video": case["spec"]["video"],
            "score_source": case["score_source"], "score_sha256": case["score_sha256"],
            "agent_source": case["agent_source"], "agent_sha256": case["agent_sha256"],
            "feature_source": case["feature_source"], "feature_sha256": case["feature_sha256"],
            "ground_truth_intervals": case["ground_truth_intervals"],
            "n_segments": case["curves"]["n_segments"], "curve_alignment": case["curves"]["align_mode"],
            "snapshots": [{key: value for key, value in snapshot.items() if key != "frame_path"}
                          for snapshot in case["snapshots"]]}


def save_figure(canvas, stem, title, cases):
    canvas.save(stem.with_suffix(".png"), dpi=(EXPORT_DPI, EXPORT_DPI))
    canvas.save(stem.with_suffix(".pdf"), resolution=EXPORT_DPI, title=title, author="VAD-Agent")
    arrays = {}
    for case in cases:
        prefix = "" if len(cases)==1 else case["spec"]["dataset"] + "_"
        arrays.update({prefix + key: case["curves"][key] for key in ("direct", "temporal", "final")})
    np.savez_compressed(str(stem) + ".curves.npz", **arrays)
    metadata = {"id": stem.name, "print_width_inches": PRINT_WIDTH,
                "print_height_inches": canvas.height / EXPORT_DPI,
                "frame_panel_pixels": [FRAME_WIDTH, IMAGE_HEIGHT],
                "body_font_points": BODY_SIZE / EXPORT_DPI * 72, "minimum_plot_font_points": 9.5,
                "full_agent_outputs": True, "projection_mode": "stride", "gap_policy": "hold",
                "cases": [case_metadata(case) for case in cases]}
    stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {stem.with_suffix('.pdf')} ({PRINT_WIDTH:.2f} x {canvas.height / EXPORT_DPI:.2f} inches)")
    return metadata


def wide_row(snapshot, frame_width=FRAME_WIDTH):
    text_x = MARGIN + frame_width + 28
    text_width = CANVAS_WIDTH - MARGIN - text_x
    box = draw_agent_box(snapshot["agent_outputs"], text_width)
    return frame_width, text_x, box, max(box.height, IMAGE_HEIGHT + 62)


def draw_wide_row(canvas, snapshot, y, letter, row, gt):
    frame_width, text_x, box, row_height = row
    draw = ImageDraw.Draw(canvas)
    draw.text((MARGIN + 8, y), f"({letter}) Frame {snapshot['frame']}", font=FONT_SMALL, fill=(0,0,0))
    draw_frame(canvas, snapshot, MARGIN, y + 62, frame_width, gt)
    canvas.paste(box, (text_x, y))


def render_abnormal(case, out_dir):
    rows = [wide_row(snapshot) for snapshot in case['snapshots']]
    row_gap = 24
    top = 95
    curve_y = top + sum(row[3] for row in rows) + row_gap * len(rows)
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        plot = plot_comparison(case, CANVAS_WIDTH - 2*MARGIN, Path(tmp)/'curve.png', height_inches=2.75)
        canvas = Image.new('RGB', (CANVAS_WIDTH, curve_y + plot.height + 25), (248,248,248))
        ImageDraw.Draw(canvas).text((MARGIN,20),case['spec']['title'],font=FONT_TITLE,fill=(0,0,0))
        y=top
        for index,(snapshot,row) in enumerate(zip(case['snapshots'],rows)):
            draw_wide_row(canvas,snapshot,y,chr(97+index),row,case['ground_truth_intervals'])
            y+=row[3]+row_gap
        canvas.paste(plot,(MARGIN,curve_y))
    return save_figure(canvas,out_dir/case['spec']['id'],case['spec']['title'],[case])


def render_normal(case, out_dir):
    snapshot=case['snapshots'][0]
    row=wide_row(snapshot)
    top=95
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        plot=plot_comparison(case,CANVAS_WIDTH-2*MARGIN,Path(tmp)/'curve.png',height_inches=2.75)
        curve_y=top+row[3]+20
        canvas=Image.new('RGB',(CANVAS_WIDTH,curve_y+plot.height+25),(248,248,248))
        draw=ImageDraw.Draw(canvas)
        draw.text((MARGIN,20),case['spec']['title'],font=FONT_TITLE,fill=(0,0,0))
        frame_width,text_x,box,row_height=row
        draw.text((MARGIN+8,top),f"Frame {snapshot['frame']}",font=FONT_SMALL,fill=(0,0,0))
        draw_frame(canvas,snapshot,MARGIN,top+62,frame_width,case['ground_truth_intervals'])
        canvas.paste(box,(text_x,top))
        canvas.paste(plot,(MARGIN,curve_y))
    return save_figure(canvas,out_dir/case['spec']['id'],case['spec']['title'],[case])


def build_preview(out_dir, metadata):
    images=[]
    for item in metadata:
        with Image.open(out_dir/(item['id']+'.png')) as source:
            image=source.convert('RGB');image.thumbnail((740,1100),Image.Resampling.LANCZOS);images.append(image.copy())
    row_heights=[max(images[i].height,images[i+1].height) for i in (0,2)]
    preview=Image.new('RGB',(1540,sum(row_heights)+60),'white')
    y=20
    for row in range(2):
        for col in range(2):
            image=images[row*2+col];preview.paste(image,(20+col*760,y))
        y+=row_heights[row]+20
    preview.save(out_dir/'preview_contact.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--ucf-frames',type=Path,default=os.environ.get('UCF_FRAMES'))
    parser.add_argument('--xd-frames',type=Path,default=os.environ.get('XD_FRAMES'))
    parser.add_argument('--out-dir',type=Path)
    args=parser.parse_args()
    if not args.ucf_frames or not args.xd_frames:
        parser.error('Set --ucf-frames and --xd-frames, or UCF_FRAMES and XD_FRAMES.')
    root=args.project_root.resolve();sys.path.insert(0,str(root))
    raw_curves=importlib.import_module('scripts.plot_raw_auc_qualitative_curves');raw_curves.ROOT=root
    out_dir=(args.out_dir or root/'figures/qualitative/paper').resolve();out_dir.mkdir(parents=True,exist_ok=True)
    frame_roots={'ucf':args.ucf_frames.resolve(),'xd':args.xd_frames.resolve()}
    cases=[load_case(spec,root,frame_roots[spec['dataset']],raw_curves) for spec in SPECS]
    metadata=[render_abnormal(case,out_dir) for case in cases[:2]]
    metadata.extend(render_normal(case,out_dir) for case in cases[2:])
    build_preview(out_dir,metadata)
    (out_dir/'figures.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()


