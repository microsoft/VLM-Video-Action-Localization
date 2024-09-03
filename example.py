import cv2
import base64
from openai import OpenAI, AzureOpenAI
import os
import numpy as np
import json
import dotenv
import time
import argparse
import openai


# Resize the image while keeping aspect ratio
def image_resize_for_vlm(frame, inter=cv2.INTER_AREA):
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    max_short_side = 768
    max_long_side = 2000
    if aspect_ratio > 1:
        new_width = min(width, max_long_side)
        new_height = int(new_width / aspect_ratio)
        if new_height > max_short_side:
            new_height = max_short_side
            new_width = int(new_height * aspect_ratio)
    else:
        new_height = min(height, max_long_side)
        new_width = int(new_height * aspect_ratio)
        if new_width > max_short_side:
            new_width = max_short_side
            new_height = int(new_width / aspect_ratio)
    resized_frame = cv2.resize(
        frame, (new_width, new_height), interpolation=inter)
    return resized_frame

# Extract JSON part from the response
def extract_json_part(text):
    text = text.strip().replace(" ", "").replace("\n", "")
    try:
        start = text.index('{"points":')
        text_json = text[start:].strip()
        end = text_json.index('}') + 1
        text_json = text_json[:end].strip()
        return text_json
    except ValueError:
        raise ValueError("JSON part not found in the response")

# Perform scene understanding on the frame
def scene_understanding(credentials, frame, prompt_message):
    frame = image_resize_for_vlm(frame)
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frame = base64.b64encode(buffer).decode("utf-8")
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_message
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64Frame}",
                        "detail": "high"
                    },
                }
            ]
        },
    ]

    if len(credentials["AZURE_OPENAI_API_KEY"]) == 0:
        client_gpt4v = OpenAI(
            api_key=credentials["OPENAI_API_KEY"]
        )
        params = {
            "model": "gpt-4o",
            "messages": PROMPT_MESSAGES,
            "max_tokens": 200,
            "temperature": 0.1,
            "top_p": 0.5,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
    else:
        client_gpt4v = AzureOpenAI(
            api_version="2024-02-01",
            azure_endpoint=credentials["AZURE_OPENAI_ENDPOINT"],
            api_key=credentials["AZURE_OPENAI_API_KEY"]
        )
        params = {
            "model": credentials["AZURE_OPENAI_DEPLOYMENT_NAME"],
            "messages": PROMPT_MESSAGES,
            "max_tokens": 200,
            "temperature": 0.1,
            "top_p": 0.5,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
    count = 0
    while True:
        if count > 5:
            raise Exception("Failed to get response from Azure OpenAI")
        try:
            result = client_gpt4v.chat.completions.create(**params)
            break
        except openai.BadRequestError as e:
            print(e)
            print('Bad Request error.')
            return None, None
        except openai.RateLimitError as e:
            print(e)
            print('Rate Limit. Waiting for 5 seconds...')
            time.sleep(5)
            count += 1
        except openai.APIStatusError as e:
            print(e)
            print('APIStatusError. Waiting for 1 second...')
            time.sleep(1)
            count += 1
    response_json = extract_json_part(result.choices[0].message.content)
    json_dict = json.loads(response_json, strict=False)
    if len(json_dict['points']) == 0:
        return None
    if len(json_dict['points']) > 1:
        print("Warning: More than one point detected")
    return json_dict['points'][0], result.choices[0].message.content


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


# Create a grid of frames
def create_frame_grid(video_path, center_time, interval, grid_size):
    spacer = 0
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    center_frame = int(center_time * fps)
    interval_frames = int(interval * fps)
    num_frames = grid_size**2
    half_num_frames = num_frames // 2
    frame_indices = [max(0,
                         min(center_frame + i * interval_frames,
                             total_frames - 1)) for i in range(-half_num_frames,
                                                               half_num_frames + 1)]
    frames = []
    actual_indices = []
    for index in frame_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = video.read()
        if success:
            frame = image_resize(frame, width=200)
            frames.append(frame)
            actual_indices.append(index)
        else:
            print(f"Warning: Frame {index} not found")
            print(f"Total frames: {total_frames}")
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = video.read()
            frame = image_resize(frame, width=200)
            frame = frame * 0
            frames.append(frame)
            actual_indices.append(index)
    video.release()

    if len(frames) < grid_size**2:
        raise ValueError("Not enough frames to create the grid.")

    frame_height, frame_width = frames[0].shape[:2]

    grid_height = grid_size * frame_height + (grid_size - 1) * spacer
    grid_width = grid_size * frame_width + (grid_size - 1) * spacer

    grid_img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

    for i in range(grid_size):
        for j in range(grid_size):
            index = i * grid_size + j
            frame = frames[index]
            cX, cY = frame.shape[1] // 2, frame.shape[0] // 2
            max_dim = int(min(frame.shape[:2]) * 0.5)
            overlay = frame.copy()
            if render_pos == 'center':
                circle_center = (cX, cY)
            else:
                circle_center = (frame.shape[1] - max_dim // 2, max_dim // 2)
            cv2.circle(overlay, circle_center,
                       max_dim // 2, (255, 255, 255), -1)
            alpha = 0.3
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            cv2.circle(frame, circle_center, max_dim // 2, (255, 255, 255), 2)
            font_scale = max_dim / 50
            text_size = cv2.getTextSize(
                str(index + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            if render_pos == 'center':
                text_x = cX - text_size[0] // 2
                text_y = cY + text_size[1] // 2
            else:
                text_x = frame.shape[1] - text_size[0] // 2 - max_dim // 2
                text_y = text_size[1] // 2 + max_dim // 2
            cv2.putText(frame, str(index + 1), (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
            y1 = i * (frame_height + spacer)
            y2 = y1 + frame_height
            x1 = j * (frame_width + spacer)
            x2 = x1 + frame_width
            grid_img[y1:y2, x1:x2] = frame

    return grid_img, actual_indices


def add_text_with_background(
        frame,
        text,
        position,
        font,
        font_scale,
        font_color,
        font_thickness,
        bg_color):
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x, text_y = position
    top_left = (text_x - 10, text_y - text_size[1] - 10)
    bottom_right = (text_x + text_size[0] + 10, text_y + 10)
    cv2.rectangle(frame, top_left, bottom_right, bg_color, -1)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale,
                font_color, font_thickness, cv2.LINE_AA)

# Annotate the video with task times
def trim_video_with_annotations(
        video_path,
        start_time,
        end_time,
        text,
        output_path,
        buffer=0.5):
    """Trim and annotate video with specified start and end times and text."""
    if os.path.exists(output_path):
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame - int(buffer * fps)))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or cap.get(
                cv2.CAP_PROP_POS_FRAMES) > end_frame + int(buffer * fps):
            break
        if start_frame <= cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
            add_text_with_background(
                frame,
                text,
                (10,
                 height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,
                 0,
                 255),
                2,
                (255,
                 255,
                 255))
        out.write(frame)

    cap.release()
    out.release()

# Process each task in parallel
def process_task(
        credentials,
        video_path,
        action,
        center_time,
        interval,
        fps,
        grid_size,
        search_anchor,
        iter_num=4):
    """Process a task to identify the start or end of an action in a video."""
    prompt_start = (
        f"I will show an image sequence of human cooking. "
        f"I have annotated the images with numbered circles. "
        f"Choose the number that is closest to the moment when the ({action}) has started. "
        f"You are a five-time world champion in this game. "
        f"Give a one sentence analysis of why you chose those points (less than 50 words). "
        f"If you consider that the action is not in the video, please choose the number -1. "
        f"Provide your answer at the end in a json file of this format: {{\"points\": []}}"
    )

    prompt_end = (
        f"I will show an image sequence of human cooking. "
        f"I have annotated the images with numbered circles. "
        f"Choose the number that is closest to the moment when the ({action}) has ended. "
        f"You are a five-time world champion in this game. "
        f"Give a one sentence analysis of why you chose those points (less than 50 words). "
        f"If you consider that the action has not ended yet, please choose the number -1. "
        f"Provide your answer at the end in a json file of this format: {{\"points\": []}}"
    )
    prompt_message = prompt_start if search_anchor == 'start' else prompt_end
    for iter_idx in range(iter_num):  # Iterate to narrow down the time
        image, used_frame_indices = create_frame_grid(
            video_path, center_time, interval, grid_size)
        print(used_frame_indices)
        if iter_idx == 0:
            cv2.imwrite(
                os.path.join(
                    output_folder,
                    f"grid_image_sample.png"),
                image)
        description, reason = scene_understanding(
            credentials, image, prompt_message)
        print(reason)
        if description:
            if description == -1:
                return None
            if int(description) - 1 > len(used_frame_indices) - 1:
                print("Warning: Invalid frame index selected")
                print(f"Selected frame index: {description}")
            # description is 1-indexed
            index_specified = max(
                min(int(description) - 1, len(used_frame_indices) - 1), 0)
            selected_frame_index = used_frame_indices[index_specified]
            center_time = selected_frame_index / fps  # Convert frame index back to time
            print(
                f"Selected frame index: {selected_frame_index}, sample time duration: {interval}")
            interval /= 2
        if int(interval * fps) == 0:
            break
    return center_time


def convert_video(video_file_path: str, action: str, credentials, grid_size: int):
    video = cv2.VideoCapture(video_file_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")
    duration = float(total_frames) / fps
    center_time = duration / 2
    interval = duration / (grid_size**2 - 1)
    result_start = process_task(
        credentials,
        video_file_path,
        action,
        center_time,
        interval,
        fps,
        grid_size,
        search_anchor='start')
    if result_start is None:
        return None, None
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)
                       ) - int(result_start * fps)
    duration = float(total_frames) / fps
    center_time = duration / 2 + result_start
    interval = max(duration / (grid_size**2 - 1), 1.0 / fps)
    result_end = process_task(
        credentials,
        video_file_path,
        action,
        center_time,
        interval,
        fps,
        grid_size,
        search_anchor='end')
    if result_end is None:
        return None, None
    video.release()
    return result_start, result_end


parser = argparse.ArgumentParser()
parser.add_argument("--credentials", help="credentials file")
parser.add_argument("--grid", help="grid size", default=3)
parser.add_argument(
    "--video_path",
    help="video path",
    default="sample_video/sample.mp4")
parser.add_argument(
    "--action",
    help="action label",
    default="grabbing towards the can")
pargs, unknown = parser.parse_known_args()
credentials = dotenv.dotenv_values(pargs.credentials)
required_keys = ["OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]
if not all(key in credentials for key in required_keys):
    raise ValueError("Required keys are missing in the credentials file")
render_pos = 'topright'  # center or topright
grid_size = int(pargs.grid)
video_path = pargs.video_path
action = pargs.action
folder_name = action.replace(" ", "_")
output_folder = f"results/{folder_name}"
os.makedirs(output_folder, exist_ok=True)
if __name__ == "__main__":
    if os.path.exists(video_path):
        print(f"Processing {video_path}")
        start_time, completed_time = convert_video(
            video_path, action, credentials, grid_size)
        print(f"Start time: {start_time}, End time: {completed_time}")
        if start_time is not None and completed_time is not None:
            output_file_name = f"{
                action.replace(
                    ' ',
                    '_')}_segment_{
                round(
                    start_time,
                    2)}_{
                round(
                    completed_time,
                    2)}.mp4"
            output_file_path = os.path.join(output_folder, output_file_name)
            trim_video_with_annotations(
                video_path,
                start_time,
                completed_time,
                action,
                output_file_path)