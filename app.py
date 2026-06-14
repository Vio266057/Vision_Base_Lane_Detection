import subprocess
import shutil
from pathlib import Path

import cv2
import numpy as np
import imageio_ffmpeg
import streamlit as st

from src.lane_inference import process_video


st.set_page_config(
    page_title="Vision-Based Lane Detection",
    page_icon="🚗",
    layout="wide",
)

MODEL_PATH = Path("models/best_unet_lane.pth")
SAMPLE_VIDEO_PATH = Path("sample_videos/road_sample.mp4")

TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

INPUT_VIDEO_PATH = TEMP_DIR / "input_video.mp4"
RAW_OUTPUT_VIDEO_PATH = TEMP_DIR / "lane_detection_raw.mp4"
DISPLAY_OUTPUT_VIDEO_PATH = TEMP_DIR / "lane_detection_output_h264.mp4"

INPUT_IMAGE_PATH = TEMP_DIR / "input_image.jpg"
IMAGE_AS_VIDEO_PATH = TEMP_DIR / "input_image_as_video.mp4"
RAW_IMAGE_OUTPUT_VIDEO_PATH = TEMP_DIR / "image_detection_raw.mp4"
OUTPUT_IMAGE_PATH = TEMP_DIR / "lane_detection_output.png"


def remove_file(path):
    path = Path(path)
    if path.exists():
        path.unlink()


def convert_to_browser_mp4(input_path, output_path):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    input_path = Path(input_path)
    output_path = Path(output_path)

    remove_file(output_path)

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

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    if not output_path.exists():
        raise FileNotFoundError(f"Converted video not found: {output_path}")
    if output_path.stat().st_size < 1000:
        raise RuntimeError("Converted video file is too small. Conversion failed.")

    return output_path


def decode_uploaded_image(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    image_array = cv2.imdecode(
        np.frombuffer(file_bytes, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image_array is None:
        raise ValueError("The uploaded image could not be decoded.")
    return image_array


def image_to_single_frame_video(image, output_path, fps=1.0):
    output_path = Path(output_path)
    remove_file(output_path)

    height, width = image.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError("Could not create a temporary video from the image.")

    writer.write(image)
    writer.release()

    if not output_path.exists():
        raise FileNotFoundError(f"Temporary image video not found: {output_path}")

    return output_path


def extract_first_frame(video_path, output_image_path):
    video_path = Path(video_path)
    output_image_path = Path(output_image_path)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open processed video: {video_path}")

    success, frame = capture.read()
    capture.release()

    if not success or frame is None:
        raise RuntimeError("Could not extract the processed image frame.")

    remove_file(output_image_path)
    if not cv2.imwrite(str(output_image_path), frame):
        raise RuntimeError(f"Could not save output image: {output_image_path}")

    return output_image_path


st.title("🚗 Vision-Based Lane Detection System")
st.write(
    "This demo uses a trained U-Net semantic segmentation model "
    "to detect lane markings from road images and video frames."
)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

st.sidebar.header("Demo Settings")
st.sidebar.write("Model path:")
st.sidebar.code(str(MODEL_PATH))
st.sidebar.info(
    "For better results, use a forward-facing road image or video "
    "with clearly visible lane markings."
)

image_tab, video_tab = st.tabs([
    "🖼️ Image Detection",
    "🎥 Video Detection",
])

with image_tab:
    st.subheader("Upload a Road Image")

    uploaded_image = st.file_uploader(
        "Choose a JPG, JPEG, or PNG road image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader",
    )

    if uploaded_image is None:
        st.info("Upload a road image to start lane detection.")
    else:
        try:
            input_image = decode_uploaded_image(uploaded_image)
            cv2.imwrite(str(INPUT_IMAGE_PATH), input_image)

            input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            st.image(
                input_image_rgb,
                caption="Uploaded Road Image",
                use_container_width=True,
            )

            if st.button("Run Image Lane Detection", key="run_image_detection"):
                with st.spinner("Processing image... Please wait."):
                    remove_file(IMAGE_AS_VIDEO_PATH)
                    remove_file(RAW_IMAGE_OUTPUT_VIDEO_PATH)
                    remove_file(OUTPUT_IMAGE_PATH)

                    image_to_single_frame_video(
                        image=input_image,
                        output_path=IMAGE_AS_VIDEO_PATH,
                    )

                    processed_video_path, frame_count = process_video(
                        input_video=IMAGE_AS_VIDEO_PATH,
                        output_video=RAW_IMAGE_OUTPUT_VIDEO_PATH,
                        model_path=MODEL_PATH,
                    )

                    output_image_path = extract_first_frame(
                        video_path=processed_video_path,
                        output_image_path=OUTPUT_IMAGE_PATH,
                    )

                    output_image_bgr = cv2.imread(str(output_image_path))
                    if output_image_bgr is None:
                        raise RuntimeError("The output image could not be read.")

                    output_image_rgb = cv2.cvtColor(
                        output_image_bgr,
                        cv2.COLOR_BGR2RGB,
                    )

                    st.success(
                        f"Lane detection completed. Processed {frame_count} frame."
                    )
                    st.subheader("Output Image")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(
                            input_image_rgb,
                            caption="Original Image",
                            use_container_width=True,
                        )
                    with col2:
                        st.image(
                            output_image_rgb,
                            caption="Lane Detection Result",
                            use_container_width=True,
                        )

                    image_bytes = output_image_path.read_bytes()
                    st.download_button(
                        label="Download Output Image",
                        data=image_bytes,
                        file_name="lane_detection_output.png",
                        mime="image/png",
                    )
        except Exception as error:
            st.error("Error during image lane detection.")
            st.exception(error)

with video_tab:
    st.subheader("Select or Upload a Road Video")

    input_mode = st.radio(
        "Choose input video source:",
        ["Use sample video", "Upload video"],
        horizontal=True,
        key="video_input_mode",
    )

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
        uploaded_video = st.file_uploader(
            "Upload an MP4, AVI, or MOV road video",
            type=["mp4", "avi", "mov"],
            key="video_uploader",
        )

        if uploaded_video is not None:
            with open(INPUT_VIDEO_PATH, "wb") as video_file:
                video_file.write(uploaded_video.getvalue())

            video_ready = True
            st.subheader("Input Video")
            st.video(str(INPUT_VIDEO_PATH))

    if video_ready:
        if st.button("Run Video Lane Detection", key="run_video_detection"):
            with st.spinner("Processing video... Please wait."):
                try:
                    remove_file(RAW_OUTPUT_VIDEO_PATH)
                    remove_file(DISPLAY_OUTPUT_VIDEO_PATH)

                    raw_output_path, frame_count = process_video(
                        input_video=INPUT_VIDEO_PATH,
                        output_video=RAW_OUTPUT_VIDEO_PATH,
                        model_path=MODEL_PATH,
                    )

                    display_output_path = convert_to_browser_mp4(
                        input_path=raw_output_path,
                        output_path=DISPLAY_OUTPUT_VIDEO_PATH,
                    )

                    st.success(
                        f"Lane detection completed. Processed {frame_count} frames."
                    )
                    st.subheader("Output Video")

                    video_bytes = display_output_path.read_bytes()
                    st.video(video_bytes, format="video/mp4")
                    st.download_button(
                        label="Download Output Video",
                        data=video_bytes,
                        file_name="lane_detection_output.mp4",
                        mime="video/mp4",
                    )
                except Exception as error:
                    st.error("Error during video lane detection.")
                    st.exception(error)
    else:
        st.info(
            "Select the sample video or upload a video to start lane detection."
        )

st.divider()
st.subheader("System Pipelines")

pipeline_col1, pipeline_col2 = st.columns(2)

with pipeline_col1:
    st.markdown("#### Image Pipeline")
    st.code(
        """
Input Road Image
      ↓
Temporary Single-Frame Video
      ↓
U-Net Lane Segmentation Model
      ↓
Binary Lane Mask Prediction
      ↓
Green Lane Overlay
      ↓
Output Image
        """
    )

with pipeline_col2:
    st.markdown("#### Video Pipeline")
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
