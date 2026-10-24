import cv2
import torch
import numpy as np
from ultralytics import YOLO
import gradio as gr
from moviepy.editor import VideoFileClip
from demo_utils import (
    MultiTaskModel,
    keypoints_to_heatmaps,
    scale_keypoints_to_size,
    heatmaps_to_processed_feature
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_model = YOLO('yolov8n-pose.pt')

multitask_model = MultiTaskModel(in_ch=17, num_classes=7).to(device)
multitask_model.load_state_dict(torch.load("best_joint_f_model.pth", map_location=device))
multitask_model.eval()

classes = ['bench_press', 'clean_and_jerk', 'jumping_jacks', 'pullup', 'pushup', 'situp', 'squat']


def process_video_for_web(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = "output_demo.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    all_keypoints = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = yolo_model(frame, verbose=False)

        annotated_frame = results[0].plot()

        if len(results[0].keypoints.data) > 0:
            kps = results[0].keypoints.data[0].cpu().numpy()
            all_keypoints.append(kps)
        else:
            all_keypoints.append(np.zeros((17, 3)))

        out.write(annotated_frame)

    cap.release()
    out.release()

    web_safe_output = "web_safe_demo.mp4"

    clip = VideoFileClip(output_path)
    clip.write_videofile(web_safe_output, codec="libx264", logger=None)
    clip.close()

    kps_array = np.array(all_keypoints)

    kps_scaled = scale_keypoints_to_size(kps_array, (width, height), (64, 64))
    heatmaps = keypoints_to_heatmaps(kps_scaled)
    p_map = heatmaps_to_processed_feature(heatmaps)

    visual_pmap = np.mean(p_map, axis=2).astype(np.uint8)
    visual_pmap = cv2.applyColorMap(visual_pmap, cv2.COLORMAP_JET)

    tensor_pmap = torch.from_numpy(p_map.transpose(2, 0, 1)).float() / 255.0
    tensor_pmap = tensor_pmap.unsqueeze(0).to(device)

    with torch.no_grad():
        cls_out, cnt_out = multitask_model(tensor_pmap)
        predicted_idx = torch.argmax(cls_out, dim=1).item()
        predicted_class = classes[predicted_idx]
        predicted_reps = float(cnt_out.item())

    return web_safe_output, visual_pmap, f"Exercise: {predicted_class} | Reps: {round(predicted_reps)}"

demo = gr.Interface(
    fn=process_video_for_web,
    inputs=gr.Video(label="Upload your Exercise Video"), # cors error, but works and displays initial video
    # inputs=gr.File(file_types=["video"], label="Upload your Exercise Video"),# no cors error, but without initial video display
    outputs=[
        gr.Video(label="Analyzed Video (Skeleton)"),
        gr.Image(label="Generated Heatmap (Motion History)"),
        gr.Textbox(label="Prediction Result")
    ],
    title="Machine learning enabled exercise tracker pipeline demo",
    description="Upload a video of a single person doing one exercise. The system will detect the skeleton, generate motion heatmap of joints, identify the exercise and count reps."
)

if __name__ == "__main__":
    demo.launch(share=True)