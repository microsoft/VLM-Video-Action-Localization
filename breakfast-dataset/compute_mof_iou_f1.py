import os
import json
import argparse
import cv2
import matplotlib.pyplot as plt


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Input JSON file containing estimated labels", required=True)
    parser.add_argument("--outdir", help="Output directory", default="out/visualize")
    return parser.parse_args()


def compute_tiou(pred_interval, gt_interval):
    """
    Compute the temporal Intersection over Union (tIoU) between two intervals.

    Args:
        pred_interval (tuple): (start_frame, end_frame) of the prediction.
        gt_interval (tuple): (start_frame, end_frame) of the ground truth.

    Returns:
        float: The tIoU value.
    """
    intersection = max(0, min(pred_interval[1], gt_interval[1]) - max(pred_interval[0], gt_interval[0]))
    union = max(pred_interval[1], gt_interval[1]) - min(pred_interval[0], gt_interval[0])
    return intersection / union if union > 0 else 0


def compute_map(pred_intervals, gt_intervals, tiou_thresholds):
    """
    Compute the mean Average Precision (mAP) over a set of tIoU thresholds.

    Args:
        pred_intervals (list of tuple): List of predicted intervals.
        gt_intervals (list of tuple): List of ground truth intervals.
        tiou_thresholds (list of float): List of tIoU thresholds.

    Returns:
        float: The computed mAP value.
    """
    assert len(pred_intervals) == len(gt_intervals)
    ap_values = []

    for threshold in tiou_thresholds:
        matches = []
        # Evaluate each prediction
        for pred in pred_intervals:
            match_found = False
            for gt in gt_intervals:
                tiou = compute_tiou(pred, gt)
                if tiou >= threshold:
                    matches.append((1, tiou))  # True Positive
                    match_found = True
                    break
            if not match_found:
                matches.append((0, 0))  # False Positive

        # Sort by tIoU (descending order)
        matches.sort(key=lambda x: x[1], reverse=True)
        tp_cum, fp_cum = 0, 0
        precisions = []
        recalls = []

        for match, _ in matches:
            if match == 1:
                tp_cum += 1
            else:
                fp_cum += 1
            precision = tp_cum / (tp_cum + fp_cum)
            recall = tp_cum / len(gt_intervals)
            precisions.append(precision)
            recalls.append(recall)

        # Compute AP using a simple approximation (area under the precision-recall curve)
        ap = 0.0
        for i in range(1, len(recalls)):
            ap += (recalls[i] - recalls[i - 1]) * precisions[i]
        ap_values.append(ap)

    return sum(ap_values) / len(ap_values) if ap_values else 0


def time_to_frame(time_in_seconds, fps):
    """Convert time in seconds to frame number based on fps."""
    return int(round(time_in_seconds * fps))


def create_label_array(total_frames, intervals):
    """
    Create a label array of length total_frames from a list of intervals.
    
    Each interval is assigned a unique label (based on its index). Every frame
    in the interval (inclusive) is assigned that label.

    Args:
        total_frames (int): Total number of frames.
        intervals (list of tuple): List of intervals (start_frame, end_frame).

    Returns:
        list: An array of labels for each frame.
    """
    labels = [-1] * total_frames
    for idx, (start, end) in enumerate(intervals):
        for frame in range(start, end + 1):
            labels[frame] = idx
    return labels


def compute_metrics(video_data, fps):
    """
    Compute various evaluation metrics (MoF, IoU per class, mean IoU, and F1 per class) 
    for a single video's predictions.

    Assumes that ground truth time intervals are 1-indexed and converts them to 0-indexed.

    Args:
        video_data (dict): Dictionary containing ground truth and predicted data.
        fps (float): Frames per second of the video.

    Returns:
        tuple: (MoF, IoU per class, mean IoU, F1 per class, mean F1)
    """
    gt_actions = video_data['action']
    gt_intervals = video_data['gt_time']
    pred_start_times = video_data['start_times']
    pred_end_times = video_data['completed_times']

    # Convert ground truth intervals from 1-indexed to 0-indexed
    gt_intervals = [(start - 1, end - 1) for start, end in gt_intervals]
    total_frames = gt_intervals[-1][1] + 1

    # Create ground truth label array
    label_gt = create_label_array(total_frames, gt_intervals)

    # Create predicted label array (initialized with -1)
    label_pred = [-1] * total_frames
    pred_keys = list(pred_start_times.keys())
    for idx, key in enumerate(pred_keys):
        start_time = pred_start_times[key]
        end_time = pred_end_times[key]
        start_frame = time_to_frame(start_time, fps)
        end_frame = time_to_frame(end_time, fps)
        for frame in range(start_frame, end_frame):
            if frame < total_frames:
                label_pred[frame] = idx

    # Fill any leading -1 values with 0
    for i in range(total_frames):
        if label_pred[i] == -1:
            label_pred[i] = 0
        else:
            break

    # Fill trailing -1 values with the last action's index
    last_index = len(pred_keys) - 1
    for i in range(total_frames - 1, -1, -1):
        if label_pred[i] == -1:
            label_pred[i] = last_index
        else:
            break

    # Ensure no -1 values remain
    if -1 in label_gt or -1 in label_pred:
        raise ValueError("Label array contains unassigned frames.")

    # Calculate Mean over Frames (MoF)
    correct_frames = sum(1 for gt, pred in zip(label_gt, label_pred) if gt == pred)
    mof = correct_frames / total_frames if total_frames > 0 else 0

    # Calculate IoU and F1 per action class
    iou_per_class = {}
    f1_per_class = {}
    for idx, action in enumerate(gt_actions):
        gt_count = sum(1 for label in label_gt if label == idx)
        pred_count = sum(1 for label in label_pred if label == idx)
        intersection = sum(1 for gt, pred in zip(label_gt, label_pred) if gt == pred == idx)
        union = gt_count + pred_count - intersection
        iou = intersection / union if union > 0 else 0
        iou_per_class[action] = iou

        tp = intersection
        fp = pred_count - intersection
        fn = gt_count - intersection
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_per_class[action] = f1

    mean_iou = sum(iou_per_class.values()) / len(iou_per_class) if iou_per_class else 0
    mean_f1 = sum(f1_per_class.values()) / len(f1_per_class) if f1_per_class else 0

    return mof, iou_per_class, mean_iou, f1_per_class, mean_f1


def process_videos(label_data_estimates, tiou_thresholds):
    """
    Process each video's data, compute evaluation metrics, and collect statistics.

    Args:
        label_data_estimates (list): List of video annotation dictionaries.
        tiou_thresholds (list): List of tIoU thresholds for mAP calculation.

    Returns:
        dict: A dictionary with per-video metrics.
        dict: A dictionary containing lists of overall metrics for plotting.
    """
    mof_list = []
    miou_list = []
    mf1_list = []
    map_list = []
    action_steps = []
    action_frames = []
    results = {}

    for video_entry in label_data_estimates:
        video_path = video_entry['video_path']

        fps = 15.0

        # Skip entries where start_times is a string (invalid data)
        if isinstance(video_entry.get('start_times'), str):
            print("Skipping video:", video_path)
            continue

        mof, iou_per_class, mean_iou, f1_per_class, mean_f1 = compute_metrics(video_entry, fps)
        mof_list.append(mof)
        miou_list.append(mean_iou)
        mf1_list.append(mean_f1)
        results[video_path] = {"MoF": mof, "mIoU": mean_iou, "mF1": mean_f1}

        # Compute durations for predicted actions
        pred_start_times = video_entry['start_times']
        pred_end_times = video_entry['completed_times']
        durations = [pred_end_times[key] - pred_start_times[key] for key in pred_start_times.keys()]
        results[video_path]["duration"] = durations

        # Convert predicted intervals to frames and compute mAP
        pred_intervals = [
            (
                time_to_frame(pred_start_times[key], fps),
                time_to_frame(pred_end_times[key], fps)
            )
            for key in pred_start_times.keys()
        ]
        gt_intervals = video_entry['gt_time']
        map_value = compute_map(pred_intervals, gt_intervals, tiou_thresholds)
        map_list.append(map_value)

        action_steps.append(len(video_entry['action']))

        # Compute total frames from ground truth (adjust for 0-index)
        gt_intervals_zero_indexed = [(start - 1, end - 1) for start, end in video_entry['gt_time']]
        total_frames = gt_intervals_zero_indexed[-1][1] + 1
        action_frames.append(total_frames)

    metrics = {
        "MoF": mof_list,
        "mIoU": miou_list,
        "mF1": mf1_list,
        "mAP": map_list,
        "action_steps": action_steps,
        "action_frames": action_frames
    }
    return results, metrics


def plot_metrics(metrics, output_path):
    """
    Generate scatter plots for the evaluation metrics and save the figure.

    Args:
        metrics (dict): Dictionary containing lists of metrics.
        output_path (str): Path to save the output plot image.
    """
    plt.figure(figsize=(12, 6))

    # Plot metrics against action steps
    ax1 = plt.subplot(2, 4, 1)
    plt.scatter(metrics["action_steps"], metrics["MoF"], alpha=0.6)
    plt.xlabel('Action Length (steps)')
    plt.ylabel('MoF')
    plt.ylim(0, 1)
    plt.title('Action Length vs MoF')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    ax2 = plt.subplot(2, 4, 2)
    plt.scatter(metrics["action_steps"], metrics["mIoU"], alpha=0.6)
    plt.xlabel('Action Length (steps)')
    plt.ylabel('mIoU')
    plt.ylim(0, 1)
    plt.title('Action Length vs mIoU')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    ax3 = plt.subplot(2, 4, 3)
    plt.scatter(metrics["action_steps"], metrics["mF1"], alpha=0.6)
    plt.xlabel('Action Length (steps)')
    plt.ylabel('mF1')
    plt.ylim(0, 1)
    plt.title('Action Length vs mF1')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    ax4 = plt.subplot(2, 4, 4)
    plt.scatter(metrics["action_steps"], metrics["mAP"], alpha=0.6)
    plt.xlabel('Action Length (steps)')
    plt.ylabel('mAP')
    plt.ylim(0, 1)
    plt.title('Action Length vs mAP')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    # Plot metrics against action frames
    ax5 = plt.subplot(2, 4, 5)
    plt.scatter(metrics["action_frames"], metrics["MoF"], alpha=0.6)
    plt.xlabel('Action Length (frames)')
    plt.ylabel('MoF')
    plt.ylim(0, 1)
    plt.title('Frames vs MoF')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    ax6 = plt.subplot(2, 4, 6)
    plt.scatter(metrics["action_frames"], metrics["mIoU"], alpha=0.6)
    plt.xlabel('Action Length (frames)')
    plt.ylabel('mIoU')
    plt.ylim(0, 1)
    plt.title('Frames vs mIoU')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    ax7 = plt.subplot(2, 4, 7)
    plt.scatter(metrics["action_frames"], metrics["mF1"], alpha=0.6)
    plt.xlabel('Action Length (frames)')
    plt.ylabel('mF1')
    plt.ylim(0, 1)
    plt.title('Frames vs mF1')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    ax8 = plt.subplot(2, 4, 8)
    plt.scatter(metrics["action_frames"], metrics["mAP"], alpha=0.6)
    plt.xlabel('Action Length (frames)')
    plt.ylabel('mAP')
    plt.ylim(0, 1)
    plt.title('Frames vs mAP')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    args = parse_arguments()
    tiou_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    input_filename = args.file

    # Load estimated label data from JSON
    with open(input_filename, "r") as f:
        label_data_estimates = json.load(f)

    # Process each video entry to compute metrics
    results, metrics = process_videos(label_data_estimates, tiou_thresholds)

    # Prepare output directories
    out_dir = args.outdir
    base_filename = os.path.splitext(os.path.basename(input_filename))[0]
    parent_dir = os.path.basename(os.path.dirname(input_filename))
    output_dir = os.path.join(out_dir, parent_dir)
    os.makedirs(output_dir, exist_ok=True)
    plot_output_path = os.path.join(output_dir, base_filename + ".png")

    # Plot and save the evaluation metrics
    plot_metrics(metrics, plot_output_path)

    # Print mean metric values
    mean_mof = sum(metrics["MoF"]) / len(metrics["MoF"]) if metrics["MoF"] else 0
    mean_miou = sum(metrics["mIoU"]) / len(metrics["mIoU"]) if metrics["mIoU"] else 0
    mean_mf1 = sum(metrics["mF1"]) / len(metrics["mF1"]) if metrics["mF1"] else 0
    mean_map = sum(metrics["mAP"]) / len(metrics["mAP"]) if metrics["mAP"] else 0

    print("Mean MoF: {:.4f}".format(mean_mof))
    print("Mean mIoU: {:.4f}".format(mean_miou))
    print("Mean mF1: {:.4f}".format(mean_mf1))
    print("Mean mAP: {:.4f}".format(mean_map))
    print("Processed videos:", len(metrics["MoF"]))

    # Save detailed results as JSON
    results_output_path = os.path.join(output_dir, base_filename + ".json")
    with open(results_output_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
