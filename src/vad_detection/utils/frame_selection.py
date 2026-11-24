# _____________________________ Frame-selection utilities used by VAD-Agent MIF_________________________________
import time
from heapq import heappush, heappop
import torch
from torch.nn.functional import normalize
from transformers import CLIPImageProcessor, CLIPVisionModel, GitVisionModel
# from transformers import BLIPImageProcessor, BLIPVisionModel
from transformers import AutoProcessor
import os
from PIL import Image

import math
from heapq import heappush, heappop



import os
import math
from PIL import Image



# selector backends (choose ONE; default is GIT)
from transformers import AutoImageProcessor, GitVisionModel  # GIT selector
device="cuda"
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14-336"
git_model_name=  "microsoft/git-base"
blip_modle_name= "Salesforce/blip-image-captioning-base"
# initialize git processors
processor = AutoProcessor.from_pretrained(git_model_name)
vision_model = GitVisionModel.from_pretrained(git_model_name).eval().to(device)
# 
# initialize clip processors
# processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_NAME)
# vision_model = CLIPVisionModel.from_pretrained(CLIP_MODEL_NAME).eval().to(device)

# with local Minima 
import torch
from heapq import heappush, heappop
from torch.nn.functional import normalize

def sample_representative_frames_MIF_GITfeat(
    frames,
    model,
    # processor,
    device="cuda",
    K=16,
    W=8,
    alpha=2.5,
    CHUNK_SIZE=512,
    return_feats=False,
):
    """
    Select frames with the *lowest* local average similarity (local minima),
    i.e., frames most distinct from their temporal neighbors (transitions/anomalies).
    Returns sorted frame indices.
    """

    # ---- Preprocess to pixel values on device ----
    inputs = processor(images=list(frames), return_tensors="pt")
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    frames = pixel_values  # [N, C, H, W]

    # ---- Adaptive window (if requested) ----
    if W == -1:
        W = max(1, int(len(frames) // (alpha * K)))
        print("window size =", W)
    else:
        W = max(1, int(W))

    # ---- Extract embeddings in chunks ----
    feat_chunks = []
    num_frames = frames.size(0)
    if num_frames == 0:
        return ([], torch.empty((0, 0), dtype=torch.float32)) if return_feats else []

    num_chunks = num_frames // CHUNK_SIZE + (1 if (num_frames % CHUNK_SIZE) > 0 else 0)

    for i in range(num_chunks):
        batch = frames[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
        with torch.no_grad():
            model_out = model(batch)

        if hasattr(model_out, "pooler_output"):
            chunk_feats = model_out.pooler_output
        else:
            # fallback: mean-pool last_hidden_state
            chunk_feats = model_out.last_hidden_state.mean(dim=1)

        chunk_feats = normalize(chunk_feats.detach(), dim=1)
        feat_chunks.append(chunk_feats)

    if len(feat_chunks) == 0:
        return ([], torch.empty((0, 0), dtype=torch.float32)) if return_feats else []

    all_feats = torch.cat(feat_chunks, dim=0)  # (N, D)
    N = all_feats.size(0)

    # ---- Similarity matrix & local average similarity (dominance) ----
    # sim in [-1, 1] since feats are L2-normalized
    all_sims = all_feats @ all_feats.transpose(0, 1)  # (N, N)

    lcl_avg = torch.full((N,), float("inf"), device=all_sims.device)
    # compute only where the window is valid
    for i in range(W, N - W):
        subsim = all_sims[i, i - W:i + W + 1]
        # exclude self-sim (the center)
        lcl_avg[i] = (subsim.sum() - 1.0) / (subsim.numel() - 1)

    # ---- Select local minima: start from the global minimum ----
    # NOTE: smaller lcl_avg => less similar to neighbors => more "anomalous"
    top_idx = int(torch.argmin(lcl_avg).item())
    res = [top_idx]

    # we'll reuse the original heap/interval logic, but rank by the *minimum* lcl_avg.
    # since heapq is a min-heap, we push negative values so that popping yields the smallest lcl_avg region.
    intvs = []
    if top_idx - W > 0:
        # get smallest in [0, top_idx - W)
        v, idx = (-lcl_avg[0:top_idx - W]).topk(1)  # max of negative == min of original
        heappush(intvs, (-v, (0, top_idx - W), idx))  # store negative score to keep compatibility
    if top_idx + W < N:
        v, idx = (-lcl_avg[top_idx + W:]).topk(1)
        heappush(intvs, (-v, (top_idx + W, N), top_idx + W + idx))

    # ---- Greedily pick remaining minima with spacing >= W ----
    while len(res) < K and len(intvs) > 0:
        top = heappop(intvs)
        top_idx = int(top[2][0].item())
        res.append(top_idx)
        l, r = top[1]

        left = top_idx - W
        right = top_idx + W

        if left > l:
            v, idx = (-lcl_avg[l:left]).topk(1)
            heappush(intvs, (-v, (l, left), l + idx))
        if right < r:
            v, idx = (-lcl_avg[right:r]).topk(1)
            heappush(intvs, (-v, (right, r), right + idx))

    selected = sorted(set(res))
    all_feats_cpu = all_feats.detach().cpu() if return_feats else None

    # ---- Cleanup large tensors from GPU ----
    del all_sims, lcl_avg
    if not return_feats:
        del all_feats
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    if return_feats:
        del all_feats
        return selected, all_feats_cpu

    return selected


def sample_representative_frames_MIF(
    frames,
    model,
    device="cuda",
    K=16,
    W=8,
    alpha=2.5,
    CHUNK_SIZE=512,
    return_feats=False,
):
    """MIF frame selector entry point.

    Set return_feats=True to also return all per-frame normalized GIT features.
    """
    return sample_representative_frames_MIF_GITfeat(
        frames=frames,
        model=model,
        device=device,
        K=K,
        W=W,
        alpha=alpha,
        CHUNK_SIZE=CHUNK_SIZE,
        return_feats=return_feats,
    )


def sample_representative_frames(frames, model=vision_model, K=16, W=8, alpha=2.5,CHUNK_SIZE = 512):
    # print(frames)
    # Added_____________
    inputs = processor(images=list(frames), return_tensors="pt")
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype) 
    frames= pixel_values
    #__________
    if W == -1: # adaptive width
        W = len(frames) // (alpha * K)
        print("window size =",W)
    
    
    
    feat_chunks = []
    num_frames = frames.size(0)
    num_chunks = num_frames // CHUNK_SIZE + (1 if (num_frames % CHUNK_SIZE) > 0 else 0)
    
    for i in range(num_chunks):
        model_out = model(frames[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE])
        if hasattr(model_out, 'pooler_output'):
            chunk_feats = model_out.pooler_output
        else:
            chunk_feats = model_out.last_hidden_state.mean(dim=1)    # pooling

        chunk_feats = chunk_feats.detach()
        chunk_feats = normalize(chunk_feats)
        feat_chunks.append(chunk_feats)
    
    if len(feat_chunks) == 0:   # empty input
        return frames.new_zeros(K, 3, 224, 224)

    all_feats = torch.cat(feat_chunks, dim=0) # (N, 768)
    all_sims = all_feats @ all_feats.transpose(0, 1)  # (N, N)
    
    # filter frames
    lcl_avg = torch.zeros(all_sims.shape[0])
    for i in range(W, all_sims.shape[0]-W):
        subsim = all_sims[i][i-W:i+W]
        lcl_avg[i] = (subsim.sum()-1) / (len(subsim) - 1)
    
    # dfs-based search
    top_idx = lcl_avg.argmax()  # the one has top lcl_avg shoule be preserved----------------->>> argmax is used here 
    res = [top_idx.item()]
    intvs = []
    if top_idx - W > 0:
        v, idx = lcl_avg[0:top_idx-W].topk(1)
        # intvs.append(((0, top-W), v, idx))
        heappush(intvs, (-v, (0, top_idx-W), idx))
    if top_idx + W < len(lcl_avg):
        v, idx = lcl_avg[top_idx+W:].topk(1)
        heappush(intvs, (-v, (top_idx+W, len(lcl_avg)), top_idx+W+idx))
        
    # while len(intvs) > 0 and len(res) < K:
    while len(res) < K and len(intvs) > 0:
        top = heappop(intvs)   # pick the next dominant frame
        top_idx = top[2][0].item()
        res.append(top_idx)
        l, r = top[1]
        left = top_idx - W
        right = top_idx + W
        if left > l:
            v, idx = lcl_avg[l:left].topk(1)
            heappush(intvs, (-v, (l, left), l+idx))
        if right < r:
            v, idx = lcl_avg[right:r].topk(1)
            heappush(intvs, (-v, (right, r), right+idx))
    # print(frames[res])
        # move big tensors off GPU
    del all_feats, all_sims, lcl_avg
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    # print(sorted(res))
    return  sorted(res)    #frames[res]










### Original sampling code "From the sealing" #------------------------->> They use argmax( ) so their frames are the ones most represent their neighbors 
# CHUNK_SIZE = 256
# INTERVAL = 20
# def sample_representative_frames(frames, model, K=16, W=8, alpha=2.5):
#     if W == -1: # adaptive width
#         W = len(frames) // (alpha * K)
    
#     feat_chunks = []
#     num_frames = frames.size(0)
#     num_chunks = num_frames // CHUNK_SIZE + (1 if (num_frames % CHUNK_SIZE) > 0 else 0)
    
#     for i in range(num_chunks):
#         model_out = model(frames[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE])
#         if hasattr(model_out, 'pooler_output'):
#             chunk_feats = model_out.pooler_output
#         else:
#             chunk_feats = model_out.last_hidden_state.mean(dim=1)    # pooling

#         chunk_feats = chunk_feats.detach()
#         chunk_feats = normalize(chunk_feats)
#         feat_chunks.append(chunk_feats)
    
#     if len(feat_chunks) == 0:   # empty input
#         return frames.new_zeros(K, 3, 224, 224)

#     all_feats = torch.cat(feat_chunks, dim=0) # (N, 768)
#     all_sims = all_feats @ all_feats.transpose(0, 1)  # (N, N)
    
#     # filter frames
#     lcl_avg = torch.zeros(all_sims.shape[0])
#     for i in range(W, all_sims.shape[0]-W):
#         subsim = all_sims[i][i-W:i+W]
#         lcl_avg[i] = (subsim.sum()-1) / (len(subsim) - 1)
    
#     # dfs-based search
#     top_idx = lcl_avg.argmax()  # the one has top lcl_avg shoule be preserved
#     res = [top_idx.item()]
#     intvs = []
#     if top_idx - W > 0:
#         v, idx = lcl_avg[0:top_idx-W].topk(1)
#         # intvs.append(((0, top-W), v, idx))
#         heappush(intvs, (-v, (0, top_idx-W), idx))
#     if top_idx + W < len(lcl_avg):
#         v, idx = lcl_avg[top_idx+W:].topk(1)
#         heappush(intvs, (-v, (top_idx+W, len(lcl_avg)), top_idx+W+idx))
        
#     # while len(intvs) > 0 and len(res) < K:
#     while len(res) < K and len(intvs) > 0:
#         top = heappop(intvs)   # pick the next dominant frame
#         top_idx = top[2][0].item()
#         res.append(top_idx)
#         l, r = top[1]
#         left = top_idx - W
#         right = top_idx + W
#         if left > l:
#             v, idx = lcl_avg[l:left].topk(1)
#             heappush(intvs, (-v, (l, left), l+idx))
#         if right < r:
#             v, idx = lcl_avg[right:r].topk(1)
#             heappush(intvs, (-v, (right, r), right+idx))
    
#     return frames[res]

# # def sample_frames_uniform(frames, K=8):
# #     num_frames = len(frames)
# #     if num_frames <= K:
# #         print(num_frames)
# #     intv = num_frames / K
    
# #     cur_idx = int(intv // 2)
# #     sampled_frames = []
# #     for _ in range(K):
# #         sampled_frames.append(frames[cur_idx])
# #         cur_idx = int(cur_idx + intv)
    
# #     assert len(sampled_frames) == K
# #     return torch.stack(sampled_frames)







# #####################Adding the Most Dominant Frames######


# # If you prefer CLIP alignment instead of GIT, uncomment these two lines and set SELECTOR_BACKEND='clip'
# # from transformers import CLIPModel, AutoProcessor as CLIPProcessor
# # CLIP_MODEL_NAME = "openai/clip-vit-large-patch14-336"



# # ============================== NEW: selector setup ==============================
# # SELECTOR_BACKEND = "git"  # options: "git" or "clip"
# # SELECTOR_MODEL_NAME = "microsoft/git-base"  # used when SELECTOR_BACKEND == "git"

# # selector_device = "cuda" if torch.cuda.is_available() else "cpu"
# # selector_dtype = torch.bfloat16  # keeps memory low without much quality loss

# # if SELECTOR_BACKEND == "git":
# #     selector_processor = AutoImageProcessor.from_pretrained(SELECTOR_MODEL_NAME)
# #     vision_model = GitVisionModel.from_pretrained(
# #         SELECTOR_MODEL_NAME, torch_dtype=selector_dtype
# #     ).eval().to(selector_device)
# # elif SELECTOR_BACKEND == "clip":
# #     selector_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
# #     vision_model = CLIPModel.from_pretrained(
# #         CLIP_MODEL_NAME, torch_dtype=selector_dtype
# #     ).eval().to(selector_device)
# # else:
# #     raise ValueError("SELECTOR_BACKEND must be 'git' or 'clip'")

# # ============================== NEW: MIF selection utility ==============================
# # Keep CHUNK_SIZE small enough to fit your GPU
# # CHUNK_SIZE = 64
# # CHUNK_SIZE = 512



@torch.no_grad()
def select_mdf_indices_with_processor(
    pil_frames,
    model,
    processor,
    K=16,
    W=8,
    alpha=2.5,
    chunk_size=256,
    device="cuda",
    dtype=torch.bfloat16,
    backend="git",
):
    """
    Select Motion-informed frames (MIF) with coverage refinement.

    Args:
        pil_frames: list of PIL.Image frames/ total snippet length
        model: vision model (GIT or CLIP)
        processor: corresponding processor
        K: number of frames to select
        W: local window size (use -1 for adaptive)
        alpha: factor for adaptive W
    """
    print(pil_frames)
    T = len(pil_frames)
    if T == 0 or K == 0:
        return []

    # --- adaptive W ---
    if W == -1:
        W = max(1, int(T / (alpha * max(K, 1))))

    # --- encode features ---
    feat_chunks = []
    for i in range(0, T, chunk_size):
        sub = pil_frames[i:i + chunk_size]
        inputs = processor(images=sub, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)

        if backend == "git":
            out = model(pixel_values=pixel_values, output_hidden_states=False)
            feats = out.last_hidden_state.mean(dim=1)  # (B, D)
        elif backend == "clip":
            feats = model.get_image_features(pixel_values=pixel_values)
        feats = torch.nn.functional.normalize(feats, dim=1)
        feat_chunks.append(feats)

    all_feats = torch.cat(feat_chunks, dim=0)  # (T, D)
    assert all_feats.size(0) == T, "Mismatch in frame encoding"

    # --- similarity + local average ---
    sim = all_feats @ all_feats.T
    lcl_avg = torch.zeros(T, device=device, dtype=sim.dtype)
    if T > 2 * W:
        for i in range(W, T - W):
            subsim = sim[i, i - W:i + W]
            lcl_avg[i] = (subsim.sum() - 1) / max(1, (len(subsim) - 1))

    # --- MIF local-minima search (dominant peaks) ---
    top_idx = int(torch.argmax(lcl_avg).item())
    selected = [top_idx]
    intvs = []
    if top_idx - W > 0:
        v, idx = lcl_avg[0:top_idx - W].topk(1)
        heappush(intvs, (-float(v.item()), (0, top_idx - W), int(idx.item())))
    if top_idx + W < T:
        v, idx = lcl_avg[top_idx + W:].topk(1)
        heappush(intvs, (-float(v.item()), (top_idx + W, T), top_idx + W + int(idx.item())))

    while len(selected) < K and intvs:
        _, (l, r), t = heappop(intvs)
        selected.append(t)
        left, right = t - W, t + W
        if left > l:
            v, idx = lcl_avg[l:left].topk(1)
            heappush(intvs, (-float(v.item()), (l, left), l + int(idx.item())))
        if right < r:
            v, idx = lcl_avg[right:r].topk(1)
            heappush(intvs, (-float(v.item()), (right, r), right + int(idx.item())))

    # --- bin-based refinement (always run, to ensure coverage) ---
    bins = torch.linspace(0, T, steps=K + 1, dtype=torch.long)
    refined = []
    for b in range(K):
        s, e = int(bins[b].item()), int(bins[b + 1].item())
        if e - s <= 0:
            continue
        j = int(torch.argmax(lcl_avg[s:e]).item())
        refined.append(s + j)

    # # --- merge DFS dominants + bin coverage ---
    # selected = sorted(set(selected + refined))[:K]

    # # --- final clamp ---
    # selected = [min(max(0, i), T - 1) for i in selected]
    
    # return selected#-----> best results so far, however got 82 
# --- merge DFS dominants + bin coverage --- 









        # --- merge DFS dominants + bin coverage --- 
    selected = sorted(set(selected))  # keep DFS results
    if len(selected) < K:
        # only then run bin coverage
        bins = torch.linspace(0, T, steps=K + 1, dtype=torch.long)
        for b in range(K):
            s, e = int(bins[b].item()), int(bins[b + 1].item())
            if e - s <= 0:
                continue
            j = int(torch.argmax(lcl_avg[s:e]).item())
            selected.append(s + j)

        selected = sorted(set(selected))[:K]

    # final clamp
    selected = [min(max(0, i), T - 1) for i in selected]
    return selected # ---> 0.854012084149214









def iter_snippet_frames(video_dir, start, end, fps_downsample=1, return_indices=False):
    """
    Load frames between [start, end) from a video directory and optionally
    return their global frame indices.

    Args:
        video_dir (str): Folder containing extracted video frames as 000001.jpg, 000002.jpg, etc.
        start (float or int): Start frame index (inclusive)
        end (float or int): End frame index (exclusive)
        fps_downsample (int): Sampling stride (default = 1)
        return_indices (bool): If True, also return the global frame numbers.

    Returns:
        frames (list of PIL.Image)
        frame_ids (list of int) if return_indices=True
    """
    start_i = int(math.floor(start))
    end_i = int(math.ceil(end))
    frame_ids = list(range(start_i, end_i, int(max(1, fps_downsample))))

    frames = []
    for fid in frame_ids:
        fp = os.path.join(video_dir, f"{fid:06d}.jpg")
        
       
        if not os.path.exists(fp):
            continue
        # .copy() ensures the file handle closes immediately
        with Image.open(fp) as im:
            
            frames.append(im.convert("RGB").copy())

    if return_indices:
        return frames, frame_ids
    else:
        return frames




