import argparse
import os, json, sys, copy
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn_extra.cluster import KMedoids
# pip install -i https://mirrors.aliyun.com/pypi/simple scikit-learn-extra
# llava
def remove_redundant(nodes):
    """Split DFS nodes into visible nodes and nodes covered by another node.

    Each item in ``nodes`` is ``([start, end], boundary_scores)``. A node is
    redundant when a smaller node is fully contained in its temporal span.
    """
    non_redundant = []
    redundant = []
    for i, node in enumerate(nodes):
        is_redundant = False
        for j, other_node in enumerate(nodes):
            if i != j:
                # æ£€æŸ¥å½“å‰èŠ‚ç‚¹æ˜¯å¦è¢«å…¶ä»–èŠ‚ç‚¹å®Œå…¨åŒ…å«
                if other_node[0][0] >= node[0][0] and other_node[0][1] <= node[0][1]: #å…¶ä»–èŠ‚ç‚¹åœ¨è¿™ä¸ªèŠ‚ç‚¹å†…éƒ¨
                    if other_node[0][0] == node[0][0] and other_node[0][1] == node[0][1]:
                        raise ValueError
                    is_redundant = True
                    break
        if not is_redundant:
            non_redundant.append(node[0])
        else:
            redundant.append(node[0])

    return non_redundant, redundant

def hierarchical(dfs_scenes_b_all, th, frames):
    """Build coarse/fine node lists from DFS nodes using one confidence cutoff.

    Nodes whose boundary confidence is consistently high become coarse nodes;
    the rest are completed against the coarse partition to keep frame coverage.
    """
    # åˆå§‹åŒ– coarse å’Œ fine åˆ—è¡¨
    coarse = []
    fine = []

    # æ ¹æ®ç½®ä¿¡åº¦é˜ˆå€¼åˆ†ç±»
    for node in dfs_scenes_b_all:
        confidence = node[1]  # èŽ·å–ç½®ä¿¡åº¦åˆ—è¡¨
        if min(confidence) >= th:
            coarse.append(node)
        else:
            fine.append(node)

    # åŽ»å†—ä½™å¤„ç†
    coarse_,redundant_c = remove_redundant(coarse)
    assert check_scenes(coarse_, frames)

    fine_,redundant_f = remove_redundant(fine)
    fine_ = fine_completion(coarse_, fine_, frames)
    assert check_scenes(fine_, frames)
    return coarse_, fine_, redundant_c+redundant_f
def kmeans_two_clusters(boundary_score):
    """
    å¯¹ boundary_score çš„ value è¿›è¡Œ KMeans äºŒåˆ†ç»„ã€‚

    è¿”å›žå€¼å§‹ç»ˆæŒ‰ç½®ä¿¡åº¦ä»Žä½Žåˆ°é«˜æŽ’åˆ—ï¼šfine keys/scores åœ¨å‰ï¼Œ
    coarse keys/scores åœ¨åŽã€‚è¿™ä¸ªé¡ºåºè¢«ä¸»æµç¨‹ç›´æŽ¥ä¾èµ–ã€‚

    Args:
        boundary_score (dict): {key: score}

    Returns:
        tuple: (class0_keys, class1_keys, threshold)
            - class0_keys: å±žäºŽç±»åˆ«0çš„keyåˆ—è¡¨
            - class1_keys: å±žäºŽç±»åˆ«1çš„keyåˆ—è¡¨
    """
    keys = list(boundary_score.keys())
    scores = np.array(list(boundary_score.values())).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    kmeans.fit(scores)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_.flatten()

    # æŒ‰æ ‡ç­¾åˆ†ç»„key
    class0_keys = [k for k, label in zip(keys, labels) if label == 0]
    class1_keys = [k for k, label in zip(keys, labels) if label == 1]

    class0_v = [boundary_score[i] for i in class0_keys]
    class1_v = [boundary_score[i] for i in class1_keys]
    if centers[0]>centers[1]:
        return class1_keys, class0_keys, class1_v, class0_v
    else:
        return class0_keys, class1_keys, class0_v, class1_v
    # fine_keys, coarse_keys, fine_v, coarse_v

def cluster_two_groups(boundary_score, algorithm='kmedoids'):
    """
    å¯¹ boundary_score çš„ value è¿›è¡ŒäºŒåˆ†ç±»ï¼Œè¿”å›žä¸¤ç±»çš„keyåˆ—è¡¨å’Œåˆ†å‰²é˜ˆå€¼

    Args:
        boundary_score (dict): {key: score}
        algorithm (str): èšç±»ç®—æ³• ['kmedoids'|'dbscan']

    Returns:
        tuple: (class0_keys, class1_keys, class0_scores, class1_scores)
    """
    keys = list(boundary_score.keys())
    scores = np.array(list(boundary_score.values())).reshape(-1, 1)

    if algorithm == 'dbscan':
        raise NotImplementedError # å½“å‰ä¸æ”¯æŒè¯¥ç®—æ³•
        # DBSCAN å®žçŽ°
        # ç¡®ä¿è½¬æ¢ä¸ºæ ‡é‡å€¼
        # score_range = np.ptp(scores)  # ç­‰ä»·äºŽ max - min
        # eps = float(score_range * 0.8)  # æ˜¾å¼è½¬æ¢ä¸ºæµ®ç‚¹æ•°

        # # æ·»åŠ å‚æ•°æ ¡éªŒ
        # if eps <= 0:
        #     eps = 1e-6  # è®¾ç½®æœ€å°æœ‰æ•ˆå€¼
        # clustering = DBSCAN(eps=eps, min_samples=1).fit(scores)
        # labels = clustering.labels_

        # # å¤„ç†å™ªå£°ç‚¹ï¼ˆå°†å™ªå£°ç‚¹åˆ†é…åˆ°æœ€è¿‘çš„ç°‡ï¼‰
        # unique_labels = np.unique(labels[labels != -1])
        # if len(unique_labels) < 2:
        #     raise ValueError("DBSCAN failed to find two clusters")

        # # è®¡ç®—ç°‡ä¸­å¿ƒ
        # centers = [scores[labels == i].mean() for i in unique_labels]

    elif algorithm == 'kmedoids':
        # KMedoids å®žçŽ°
        kmedoids = KMedoids(n_clusters=2, init='k-medoids++', random_state=0)
        labels = kmedoids.fit_predict(scores)
        centers = kmedoids.cluster_centers_.flatten()

    else:
        raise ValueError("Unsupported algorithm")

    # æŒ‰ä¸­å¿ƒå€¼æŽ’åºç¡®å®šç±»åˆ«é¡ºåº
    sorted_indices = np.argsort(centers)
    class0_mask = (labels == sorted_indices[0])
    class1_mask = (labels == sorted_indices[1])

    # ç”Ÿæˆç»“æžœ
    class0_keys = [k for k, mask in zip(keys, class0_mask) if mask]
    class1_keys = [k for k, mask in zip(keys, class1_mask) if mask]
    class0_scores = [boundary_score[k] for k in class0_keys]
    class1_scores = [boundary_score[k] for k in class1_keys]

    # è®¡ç®—åˆ†å‰²é˜ˆå€¼ï¼ˆä¸¤ç±»ä¸­å¿ƒçš„ä¸­å€¼ï¼‰
    threshold = np.median([max(class0_scores), min(class1_scores)])

    return class0_keys, class1_keys, class0_scores, class1_scores
def find_closest(a_list, x):
    if not a_list:
        return None  # å¤„ç†ç©ºåˆ—è¡¨çš„æƒ…å†µ
    closest = a_list[0]
    min_diff = abs(a_list[0] - x)
    for num in a_list[1:]:
        current_diff = abs(num - x)
        if current_diff < min_diff:
            min_diff = current_diff
            closest = num
    return closest

def get_idx_from_score_by_threshold(threshold=0.5, seq_indices=None, seq_scores=None):
    seq_indices = np.array(seq_indices)
    seq_scores = np.array(seq_scores)
    bdy_indices = []
    internals_indices = []
    for i in range(len(seq_scores)):
        if seq_scores[i] >= threshold:
            internals_indices.append(i)
        elif seq_scores[i] < threshold and len(internals_indices) != 0:
            bdy_indices.append(internals_indices)
            internals_indices = []

        if i == len(seq_scores) - 1 and len(internals_indices) != 0:
            bdy_indices.append(internals_indices)

    bdy_indices_in_video = []
    bdy_indices_in_video_score = []
    if len(bdy_indices) != 0:
        for internals in bdy_indices:
            center = int(np.mean(internals))
            bdy_indices_in_video.append(seq_indices[center])
            center_x = find_closest(internals, center)
            bdy_indices_in_video_score.append(round(seq_scores[center_x],4))
    return bdy_indices_in_video, bdy_indices_in_video_score

def get_peak_idx_from_score_by_threshold(threshold=0.5, seq_indices=None, seq_scores=None):
    # seq_indices = np.array(seq_indices)
    # seq_scores = np.array(seq_scores)
    peaks = []
    peaks_scores = []
    for i in range(1, len(seq_indices)-1):
        if seq_scores[i] >= threshold:
            if seq_scores[i]>=seq_scores[i-1] and seq_scores[i]>=seq_scores[i+1]:
                peaks.append(seq_indices[i])
                peaks_scores.append(seq_scores[i])
    return peaks, peaks_scores


def temporal_testing_annotations(temporal_annotation_file):
    annotations = {}

    with open(temporal_annotation_file, 'r') as annotations_f:
        for line in annotations_f:
            parts = line.strip().split()
            video_name = str(parts[0])
            annotation_values = parts[2:]
            annotations[video_name] = annotation_values

    return annotations

def calculate_iou(a,b, s,e):
    max_start = max(s, a)
    min_end = min(e, b)
    if max_start >= min_end:
        return 0, 0
    else:
        intersection = min_end - max_start

        # è®¡ç®—å„ä¸ªåŒºé—´çš„é•¿åº¦
        scene_length = e - s + 1
        input_length = b - a + 1

        # å¹¶é›†é•¿åº¦
        union = scene_length + input_length - intersection
        return intersection / union, intersection

def calculate_dfs_all_idx(frame_end, out_idx, out_idx_score, min_length):
    '''
    çˆ¶èŠ‚ç‚¹ä¹Ÿä¼šä¿ç•™
    :param frame_end:
    :param out_idx:
    :param out_idx_score:
    :param min_length:
    :return:
    '''
    # å°†è¾¹ç•Œå¸§å’Œç½®ä¿¡åº¦é…å¯¹å¹¶æŒ‰ç½®ä¿¡åº¦é™åºæŽ’åºï¼ˆç½®ä¿¡åº¦ç›¸åŒåˆ™æŒ‰å¸§å·å‡åºï¼‰
    candidates = sorted(zip(out_idx, out_idx_score), key=lambda x: (-x[1], x[0]))
    frame_score_dict = dict(candidates)
    frame_score_dict[0] = 1.
    frame_score_dict[frame_end] = 1.

    stack = [[0, frame_end]]  # åˆå§‹åŒ–æ ˆï¼Œæ ¹èŠ‚ç‚¹ä¸ºæ•´ä¸ªè§†é¢‘ï¼ˆframesï¼‰
    used = set()         # è®°å½•å·²ä½¿ç”¨çš„åˆ†å‰²ç‚¹
    segments = []        # å­˜å‚¨æœ€ç»ˆåˆ†å‰²åŽçš„åŒºé—´
    segments_b = []       # å­˜å‚¨æœ€ç»ˆåˆ†å‰²åŽçš„åŒºé—´çš„è¾¹ç•Œåˆ†æ•°

    while stack:
        start, end = stack.pop()
        segments_b.append([[start, end], [frame_score_dict[start], frame_score_dict[
            end]]])
        # length = end - start

        # å¦‚æžœå½“å‰åŒºé—´é•¿åº¦å°äºŽé˜ˆå€¼ï¼Œå…ˆåˆå¹¶
        # if length < min_length:
        #     segments.append((start, end))
        #     continue

        # å¯»æ‰¾å½“å‰åŒºé—´å†…æœªä½¿ç”¨ä¸”ç½®ä¿¡åº¦æœ€é«˜çš„åˆ†å‰²ç‚¹
        split_frame = None
        for frame, score in candidates:
            if frame in used:
                continue
            if start < frame < end:
                split_frame = frame
                used.add(frame)
                break

        # å¦‚æžœæ‰¾åˆ°åˆ†å‰²ç‚¹ï¼Œåˆ†å‰²æˆå·¦å³å­åŒºé—´
        if split_frame is not None:

            if split_frame - start < min_length  or end - split_frame<min_length: # å†åˆ†ä¼šå‡ºçŽ°å¤ªå°çš„æ®µ
                if split_frame - start >= min_length  or end - split_frame >= min_length: # æœ‰ä¸€ä¸ªå­æ®µè¿˜å¯åˆ†,é‚£ä¹ˆ
                    ### å¢žåŠ äº†ä¸€ä¸ªé€»è¾‘ï¼Œè§†é¢‘ä¸¤å¤´å‡ºçŽ°çš„åˆ†å‰²ç‚¹ä¸å› ä¸¤ç«¯é•¿åº¦é™åˆ¶è€Œä¸¢å¼ƒ
                    if start==0 and split_frame - start < min_length:
                        segments.append([start, split_frame]) # å·¦å¤´çš„å­æ®µï¼Œå¿½ç•¥é™åˆ¶è¿›è¡Œåˆ‡åˆ†
                        segments_b.append([[start, split_frame], [frame_score_dict[start], frame_score_dict[
                            split_frame]]])
                        stack.append([split_frame, end]) # å‰©ä¸‹ä¸€æ®µè‚¯å®šè¿˜èƒ½åˆ‡åˆ†

                    elif end==frame_end and end - split_frame < min_length:
                        segments.append([split_frame, end]) # å³å¤´çš„å­æ®µï¼Œå¿½ç•¥é™åˆ¶è¿›è¡Œåˆ‡åˆ†
                        segments_b.append([[split_frame, end], [frame_score_dict[split_frame], frame_score_dict[
                            end]]])
                        stack.append([start, split_frame]) # å‰©ä¸‹ä¸€æ®µè‚¯å®šè¿˜èƒ½åˆ‡åˆ†
                    else:
                        # raise NotImplementedError
                        stack.append([start, end]) # å…¶ä»–æƒ…å†µ,æš‚ä¸è¿›è¡Œåˆ‡åˆ†,å¯èƒ½å¯ä»¥æœç´¢åˆ°å…¶ä»–åˆé€‚çš„åˆ‡åˆ†ç‚¹
                        segments_b.pop(-1) # é˜²æ­¢segmentsé‡å¤
                else:
                    segments.append([start, end]) # æ‰€æœ‰å­æ®µä¸å¯åˆ†, ç›´æŽ¥åŠ å…¥åˆ†å‰²ç»“æžœ
            else: # è¿™ä¸ªåŒºé—´é•¿åº¦è¶³å¤Ÿé•¿ï¼Œè¿˜èƒ½åˆ†
                stack.append([split_frame, end])  # å³å­èŠ‚ç‚¹å…ˆå…¥æ ˆï¼ˆåŽå¤„ç†ï¼‰
                stack.append([start, split_frame]) # å·¦å­èŠ‚ç‚¹åŽå…¥æ ˆï¼ˆå…ˆå¤„ç†ï¼‰
        else: # æ²¡æœ‰å¯ç”¨åˆ†å‰²ç‚¹äº†ï¼Œç›´æŽ¥åŠ å…¥åˆ†å‰²ç»“æžœ
            segments.append([start, end])
    # æŒ‰èµ·å§‹æ—¶é—´æŽ’åºå¹¶è¿”å›žæœ€ç»ˆåˆ†å‰²åŒºé—´

    segments.sort()
    segments_b.sort(key=lambda x: x[0][0])

    return segments, segments_b

def check_scenes(dfs_scenes, frames):
    assert dfs_scenes[0][0] == 0
    assert dfs_scenes[-1][-1] == frames-1
    for i in range(len(dfs_scenes[:-1])):
        if not dfs_scenes[i][1] == dfs_scenes[i+1][0]:
            raise ValueError
    return True

def fine_completion(coarse, fine, frames):

    common = []
    if fine[0][0]!=0:
        fine.insert(0, coarse[0])
    if fine[-1][-1]!=frames-1:
        fine.append(coarse[-1])

    for i in range(len(fine)-1):
        if fine[i][1]!= fine[i+1][0]:
            for j in coarse:
                if j[0]>=fine[i][1] and j[1]<=fine[i+1][0]:
                    common.append(j)

    fine.extend(common)
    fine.sort()
    return fine

if __name__ == "__main__":
    # æž„å»ºparaser argså‚æ•°è¾“å…¥
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, 
                        default='./result/UCF_Crime_test/EGEBD_x2x3x4_r50_eff_split_out_th0.5/pred_scenes_th0.5.json',
                         help='Path to the EfficientGEBD results JSON file')

    parser.add_argument('--temporal_annotation_file', type=str, 
        default=None,
        # default='./dataset_info/ucf_crime/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt',
        help='Path to the temporal annotation file for evaluation segmentation performance, if None, auto set according to json_path'
    )

    parser.add_argument('--threshold', type=str,
                        default='kmeans', 
                        # default='kmedoids',
                        # default=0.5,
                        help='Threshold value or mode')
    parser.add_argument('--peak', type=str, default='_peak', help='Peak mode')
    parser.add_argument('--gamma', type=float, default=0.4, help='Gamma value for HGTree')
    parser.add_argument('--min_length', type=int, default=1, help='Minimum length for DFS segmentation')
    args = parser.parse_args()

    # # è‡ªåŠ¨é…ç½®ä¸åŒæ•°æ®é›†çš„å‚æ•°
    if args.temporal_annotation_file is None:
        # This file lives at the VADTree repo root, so __file__ already points
        # to the directory that contains dataset_info.
        vadtree_path = os.path.dirname(os.path.abspath(__file__))
        if 'UCF' in args.json_path:
            args.temporal_annotation_file=f'{vadtree_path}/dataset_info/ucf_crime/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt'
        elif 'XD' in args.json_path:
            args.temporal_annotation_file=f'{vadtree_path}/dataset_info/xd_violence/annotations/temporal_anomaly_annotation_for_testing_videos.txt'
        elif 'MSAD' in args.json_path:
            args.temporal_annotation_file=f'{vadtree_path}/dataset_info/msad/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt'
        else:
            raise NotImplementedError

    annotations = temporal_testing_annotations(args.temporal_annotation_file)
    json_path = args.json_path
    with open(json_path) as f:
        org_all_video_scenes_dict = json.load(f)

    # pprint(all_video_scores_dict['config'])
    # print(json.dumps(all_video_scenes_dict['config'], indent=4, ensure_ascii=False))
    # Keep anomaly videos first and normal videos last. The order is used only
    # for deterministic logging/output traversal, not for metric computation.
    sorted_keys = sorted(
        org_all_video_scenes_dict.keys(),
        key=lambda x: ("ormal" in x or 'label_A' in x, x)
    )

    all_video_scenes_dict = {k: org_all_video_scenes_dict[k] for k in sorted_keys}

    all_max_iou_I_list = {'coarse': [],  'fine':[], 'fine + coarse':[], 'nodes':[]}
    all_max_iou_list = []
    all_max_intersection_list = []

    all_gt = []
    all_coarse_scenes = []
    all_fine_scenes = []
    all_nodes = []
    all_coarse_min = []

    # EfficientGEBDçš„ç»“æžœè·¯å¾„
    json_path = args.json_path
    peak = args.peak
    threshold = args.threshold
    # threshold = 'kmedoids'

    gamma = args.gamma
    min_length = args.min_length

    assert peak in ['', '_peak']
    out_json_dir = os.path.dirname(json_path) + f'{peak}_dfs_{threshold}_{min_length}_{gamma}'

    out_pred_json_path = out_json_dir + f'/pred.json'
    out_coarse_json_path = out_json_dir + f'/dfs_coarse_scenes.json'
    out_fine_json_path = out_json_dir + f'/dfs_fine_scenes.json'
    out_redundant_json_path = out_json_dir + f'/dfs_redundant_scenes.json'
    os.makedirs(os.path.dirname(out_pred_json_path), exist_ok=True)

    dfs_pred_scenes_dict = copy.deepcopy(all_video_scenes_dict)
    dfs_coarse_scenes_dict = copy.deepcopy(all_video_scenes_dict)
    dfs_fine_scenes_dict = copy.deepcopy(all_video_scenes_dict)
    dfs_redundant_scenes_dict = copy.deepcopy(all_video_scenes_dict)
    coarse_min_list= []
    coarse_peak_num= []
    fine_peak_num= []
    for idx, (k,v) in enumerate(annotations.items()):
        # print(idx, k)
        assert 'fps' in all_video_scenes_dict[k].keys()
        fps, frames = all_video_scenes_dict[k]['fps'], all_video_scenes_dict[k]['frames']

        idx_split_100 = np.linspace(0, frames-1, int((frames/fps)*min(10, fps)), dtype=int).tolist()  #è¿˜åŽŸGEBDæ—¶é‡‡æ ·ç´¢å¼•
        print(idx, k, fps, frames)
        if threshold in ['kmeans', 'kmedoids']:
            if peak == '_peak':
                out_idx, out_idx_score = get_peak_idx_from_score_by_threshold(threshold=gamma,
                    seq_indices=idx_split_100, seq_scores=all_video_scenes_dict[k]['pred'])
            elif peak == '':
                out_idx, out_idx_score = get_idx_from_score_by_threshold(threshold=gamma, seq_indices=idx_split_100,
                    seq_scores=all_video_scenes_dict[k]['pred'])
            out_idx = [int(i) for i in out_idx]
            fine, nodes_score = calculate_dfs_all_idx(frames-1, out_idx, out_idx_score,
                min_length=min_length)
            nodes = [i[0] for i in nodes_score]
            all_boundary_score = {k: v for k, v in zip(idx_split_100, all_video_scenes_dict[k]['pred'])}
            peak_boundary_score = {k: v for k, v in zip(out_idx, out_idx_score)}

            legal_peak_idx = []
            for i in fine:
                legal_peak_idx.extend(i)
            legal_peak_idx = sorted(legal_peak_idx)[1:-1]
            # Remove artificial root/end boundaries before clustering; only real
            # candidate boundaries should decide the coarse/fine split.
            legal_peak_boundary_score = {k: peak_boundary_score[k] for k in legal_peak_idx}

            if len(set(legal_peak_boundary_score.values()))>=2:  # ä¿è¯å¯ä»¥èšç±»
                if threshold == 'kmeans':
                    # if len
                    fine_keys, coarse_keys, fine_v, coarse_v = kmeans_two_clusters(legal_peak_boundary_score)
                else:
                    fine_keys, coarse_keys, fine_v, coarse_v = cluster_two_groups(legal_peak_boundary_score,
                        algorithm=threshold)

                try:
                    assert min(coarse_v)>max(fine_v)
                except:
                    print('Not surport clustersï¼')
                    print('len(peak_boundary_score):',len(legal_peak_boundary_score))
                    coarse = fine
                    fine = fine
                    redundant = fine
                    continue
                th = min(coarse_v)
                coarse, fine, redundant = hierarchical(nodes_score, th, frames)
                print('clusters info:', th, len(coarse_keys), len(fine_keys),  f'coarse:{len(coarse)} fine:'
                                                                               f' {len(fine)}')
                print(sorted(coarse_v, reverse=True), '   ', sorted(fine_v, reverse=True))
                all_coarse_min.append(min(coarse_v))
                coarse_min_list.append(min(coarse_v))
                coarse_peak_num.append(len(coarse_keys))
                fine_peak_num.append(len(fine_keys))
            else:
                print('len(peak_boundary_score):',len(legal_peak_boundary_score))
                coarse = fine
                fine = fine
                redundant = fine
        else:
            raise NotImplementedError
            # out_idx, out_idx_score = get_peak_idx_from_score_by_threshold(threshold=threshold, seq_indices=idx_split_100,  seq_scores=all_video_scenes_dict[k]['pred'])
            # out_idx = [int(i) for i in out_idx]
            # dfs_scenes = calculate_dfs_idx(frames-1, out_idx, out_idx_score, min_length=min_length)
            # assert check_scenes(dfs_scenes, frames)

        # coarse
        dfs_coarse_scenes_dict[k]['scenes'] = coarse
        del dfs_coarse_scenes_dict[k]['pred'] #
        dfs_coarse_scenes_dict[k]['threshold'] = threshold
        dfs_coarse_scenes_dict[k]['min_length'] = min_length
        dfs_coarse_scenes_dict[k]['gamma'] = gamma

        # fine
        dfs_fine_scenes_dict[k]['scenes'] = fine
        del dfs_fine_scenes_dict[k]['pred'] # åˆ é™¤æ—§çš„
        dfs_fine_scenes_dict[k]['threshold'] = threshold
        dfs_fine_scenes_dict[k]['min_length'] = min_length
        dfs_fine_scenes_dict[k]['gamma'] = gamma

        # redundant
        dfs_redundant_scenes_dict[k]['scenes'] = redundant
        del dfs_redundant_scenes_dict[k]['pred'] # åˆ é™¤æ—§çš„
        dfs_redundant_scenes_dict[k]['threshold'] = threshold
        dfs_redundant_scenes_dict[k]['min_length'] = min_length
        dfs_redundant_scenes_dict[k]['gamma'] = gamma

        dfs_pred_scenes_dict[k] = copy.deepcopy(dfs_coarse_scenes_dict[k])  # å¿…é¡»æ·±æ‹·è´
        del dfs_pred_scenes_dict[k]['scenes'] # åˆ é™¤scenes
        dfs_pred_scenes_dict[k]['pred'] = [round(i,4) for i in all_video_scenes_dict[k]['pred']]

        all_nodes.extend(nodes)
        all_coarse_scenes.extend(coarse)
        all_fine_scenes.extend(fine)
        # assert len(nodes) >= len(coarse) + len(fine)
        # åŸºäºŽgtè¯„ä¼°åœºæ™¯åˆ†å‰²åŽä¸Žgtçš„iou, æž„å»ºHGTreeçš„è¿‡ç¨‹æœ¬èº«ä¸éœ€è¦è¯¥æ­¥éª¤
        # max_iou_list = []
        # max_intersection_list = []
        # iou_type = 'nodes'
        # iou_type = 'fine'
        # iou_type = 'coarse'
        # iou_type = 'fine + coarse'

        iou_type = {'coarse': coarse,  'fine':fine, 'fine + coarse':fine + coarse, 'nodes':fine + coarse + nodes}
        max_iou_I_list = {'coarse': [],  'fine':[], 'fine + coarse':[], 'nodes':[]}
        for i in range(0, len(v), 2):
            a_str, b_str = v[i], v[i+1]
            if a_str == '-1' or b_str == '-1':
                continue
            a, b = int(a_str), int(b_str)
            all_gt.append([a,b])
            iou_I_list = {'coarse': [],  'fine':[], 'fine + coarse':[], 'nodes':[]}
            for kk, vv in iou_type.items():
                for s, e in vv:
                    # for s,e in fine + coarse + nodes:
                    iou, intersection = calculate_iou(a, b, s, e)
                    iou_I_list[kk].append((iou, intersection))
                max_iou_I_list[kk].append(sorted(iou_I_list[kk])[-1])
                # max_intersection_list.append(max(intersection_list))
        # print(idx, k, max_iou_I_list)
        # all_max_iou_list.extend(max_iou_list)
        # all_max_intersection_list.extend(max_intersection_list)
        for kk in max_iou_I_list.keys(): all_max_iou_I_list[kk].extend(max_iou_I_list[kk])

    with open(out_pred_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(dfs_pred_scenes_dict, json_file, ensure_ascii=False, indent=4)
    with open(out_coarse_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(dfs_coarse_scenes_dict, json_file, ensure_ascii=False, indent=4)
    with open(out_fine_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(dfs_fine_scenes_dict, json_file, ensure_ascii=False, indent=4)
    with open(out_redundant_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(dfs_redundant_scenes_dict, json_file, ensure_ascii=False, indent=4)
    print(f'save result to:\n{os.path.abspath(out_pred_json_path)}, \n{os.path.abspath(out_coarse_json_path)}, \n{os.path.abspath(out_fine_json_path)}, \n{os.path.abspath(out_redundant_json_path)}')
    
    
    print('-----------------------------------------analysis result---------------------------------------------')
    print('peak:',peak, threshold, min_length,gamma , json_path)
    for kk in max_iou_I_list.keys():
        print(f'{kk}  all mIoU and mIF:', np.array(all_max_iou_I_list[kk]).mean(0).round(2))
    all_gt_len = np.array([i[1]-i[0]for i in all_gt])
    all_fine_scenes_len = np.array([i[1]-i[0]for i in all_fine_scenes])
    all_coarse_scenes_len = np.array([i[1]-i[0]for i in all_coarse_scenes])
    all_nodes_len = np.array([i[1]-i[0]for i in all_nodes])

    print(f'gt len: {all_gt_len.mean()}   all_coarse_scenes_len_mean: {all_coarse_scenes_len.mean():.2f}  '
          f'all_coarse_scenes '
          f'num:{len(all_coarse_scenes_len)}')
    print(f'gt len: {all_gt_len.mean()}   all_fine_scenes_len_mean: {all_fine_scenes_len.mean():.2f}  all_fine_scenes '
          f'num:{len(all_fine_scenes_len)}')
    print(f'gt len: {all_gt_len.mean()}   coarse+fine_scenes num:{len(all_fine_scenes_len) + len(all_coarse_scenes_len)}')
    print(f'gt len: {all_gt_len.mean()}   all_nodes_len_mean: {all_nodes_len.mean():.2f}  all_nodes '
          f'num:{len(all_nodes_len)}')

    # print(all_coarse_min) # coarse clusterçš„æœ€å°ç½®ä¿¡åº¦
    print(np.array(all_coarse_min).mean())

