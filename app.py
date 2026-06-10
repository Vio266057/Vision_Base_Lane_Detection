import subprocess
import imageio_ffmpeg
import streamlit as st
from pathlib import Path
import shutil

from src.lane_inference import process_video


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Vision-Based Lane Detection",
    page_icon="🚗",
    layout="wide",
)


# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = Path("models/best_unet_lane.pth")
SAMPLE_VIDEO_PATH = Path("sample_videos/road_sample.mp4")

TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

INPUT_VIDEO_PATH = TEMP_DIR / "input_video.mp4"
RAW_OUTPUT_VIDEO_PATH = TEMP_DIR / "lane_detection_raw.mp4"
DISPLAY_OUTPUT_VIDEO_PATH = TEMP_DIR / "lane_detection_output_h264.mp4"


def convert_to_browser_mp4(input_path, output_path):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    if output_path.exists():
        output_path.unlink()

    command = [
        ffmpeg_exe,
        "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(output_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    if not output_path.exists():
        raise FileNotFoundError(f"Converted video not found: {output_path}")

    if output_path.stat().st_size < 1000:
        raise RuntimeError("Converted video file is too small. Conversion failed.")

    return output_path


# -----------------------------
# UI Header
# -----------------------------
st.title("🚗 Vision-Based Lane Detection System")
st.write(
    "This demo uses a trained U-Net semantic segmentation model to detect lane markings from road video frames."
)


# -----------------------------
# Check Model
# -----------------------------
if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Demo Settings")

input_mode = st.sidebar.radio(
    "Choose input video source:",
    ["Use sample video", "Upload video"]
)

st.sidebar.write("Model path:")
st.sidebar.code(str(MODEL_PATH))


# -----------------------------
# Select or Upload Video
# -----------------------------
video_ready = False

if input_mode == "Use sample video":
    if SAMPLE_VIDEO_PATH.exists():
        shutil.copy(SAMPLE_VIDEO_PATH, INPUT_VIDEO_PATH)
        video_ready = True
        st.subheader("Input Video")
        st.video(str(INPUT_VIDEO_PATH))
    else:
        st.warning(f"Sample video not found: {SAMPLE_VIDEO_PATH}")

else:
    uploaded_file = st.file_uploader(
        "Upload a road video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_file is not None:
        with open(INPUT_VIDEO_PATH, "wb") as f:
            f.write(uploaded_file.read())

        video_ready = True
        st.subheader("Input Video")
        st.video(str(INPUT_VIDEO_PATH))


# -----------------------------
# Run Detection
# -----------------------------
if video_ready:
    run_button = st.button("Run Lane Detection")

    if run_button:
        with st.spinner("Processing video... Please wait."):
            try:
                # Remove old temp output files
                if RAW_OUTPUT_VIDEO_PATH.exists():
                    RAW_OUTPUT_VIDEO_PATH.unlink()

                if DISPLAY_OUTPUT_VIDEO_PATH.exists():
                    DISPLAY_OUTPUT_VIDEO_PATH.unlink()

                raw_output_path, frame_count = process_video(
                    input_video=INPUT_VIDEO_PATH,
                    output_video=RAW_OUTPUT_VIDEO_PATH,
                    model_path=MODEL_PATH,
                )

                display_output_path = convert_to_browser_mp4(
                    input_path=raw_output_path,
                    output_path=DISPLAY_OUTPUT_VIDEO_PATH,
                )

                st.success(f"Lane detection completed. Processed {frame_count} frames.")

                st.subheader("Output Video")

                video_bytes = display_output_path.read_bytes()


                st.video(video_bytes, format="video/mp4")

                st.download_button(
                    label="Download Output Video",
                    data=video_bytes,
                    file_name="lane_detection_output.mp4",
                    mime="video/mp4",
                )
            except Exception as e:
                st.error("Error during lane detection.")
                st.exception(e)

else:
    st.info("Please select or upload a video to start lane detection.")


# -----------------------------
# Project Info
# -----------------------------
st.divider()

st.subheader("System Pipeline")
st.code(
    """
Input Road Video
      ↓
Frame Extraction
      ↓
U-Net Lane Segmentation Model
      ↓
Binary Lane Mask Prediction
      ↓
Green Lane Overlay
      ↓
Output Video
    """
)