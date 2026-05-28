import os

from PIL import Image


def create_gif_from_png_frames(
    frames_dir: str,
    gif_name: str = "animation.gif",
    frame_time: float = 0.2,
    stop_time: float = 1.0,
    gif_path: str | None = None,
) -> str:
    if not os.path.isdir(frames_dir):
        raise ValueError(f"Frame directory does not exist: {frames_dir}")

    frame_names = sorted(
        file_name
        for file_name in os.listdir(frames_dir)
        if file_name.lower().endswith(".png")
    )
    if not frame_names:
        raise ValueError(f"No PNG frames found in: {frames_dir}")

    if gif_path is None:
        gif_path = os.path.join(frames_dir, gif_name)
    gif_parent = os.path.dirname(gif_path)
    if gif_parent:
        os.makedirs(gif_parent, exist_ok=True)
    frame_duration_ms = int(1000 * frame_time)
    stop_duration_ms = int(1000 * stop_time)

    frames = []
    for frame_name in frame_names:
        frame_path = os.path.join(frames_dir, frame_name)
        with Image.open(frame_path) as image:
            frames.append(image.convert("RGBA"))

    durations = [frame_duration_ms] * len(frames)
    durations[-1] = stop_duration_ms

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        disposal=2,
    )
    for frame in frames:
        frame.close()

    print(f"Saved GIF to: {gif_path}")
    return gif_path


def stack_png_frame_directories_vertically(
    top_frames_dir: str,
    bottom_frames_dir: str,
    output_dir: str,
) -> list[str]:
    if not os.path.isdir(top_frames_dir):
        raise ValueError(f"Top frame directory does not exist: {top_frames_dir}")
    if not os.path.isdir(bottom_frames_dir):
        raise ValueError(f"Bottom frame directory does not exist: {bottom_frames_dir}")

    top_frame_names = sorted(
        file_name
        for file_name in os.listdir(top_frames_dir)
        if file_name.lower().endswith(".png")
    )
    bottom_frame_names = sorted(
        file_name
        for file_name in os.listdir(bottom_frames_dir)
        if file_name.lower().endswith(".png")
    )
    if not top_frame_names:
        raise ValueError(f"No PNG frames found in: {top_frames_dir}")
    if not bottom_frame_names:
        raise ValueError(f"No PNG frames found in: {bottom_frames_dir}")
    if len(top_frame_names) != len(bottom_frame_names):
        raise ValueError(
            f"Frame count mismatch: {len(top_frame_names)} top frames vs "
            f"{len(bottom_frame_names)} bottom frames."
        )

    os.makedirs(output_dir, exist_ok=True)
    output_paths = []
    for frame_idx, (top_name, bottom_name) in enumerate(zip(top_frame_names, bottom_frame_names)):
        top_path = os.path.join(top_frames_dir, top_name)
        bottom_path = os.path.join(bottom_frames_dir, bottom_name)
        out_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")

        with Image.open(top_path) as top_image, Image.open(bottom_path) as bottom_image:
            top_image = top_image.convert("RGBA")
            bottom_image = bottom_image.convert("RGBA")
            target_width = max(top_image.width, bottom_image.width)

            def resize_to_width(image):
                if image.width == target_width:
                    return image
                scale = target_width / image.width
                target_height = int(round(image.height * scale))
                return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

            top_image = resize_to_width(top_image)
            bottom_image = resize_to_width(bottom_image)
            combined_height = top_image.height + bottom_image.height
            combined_image = Image.new("RGBA", (target_width, combined_height), (255, 255, 255, 255))
            combined_image.paste(top_image, (0, 0))
            combined_image.paste(bottom_image, (0, top_image.height))
            combined_image.save(out_path)

        output_paths.append(out_path)
        print(f"Saved combined frame to: {out_path}")

    return output_paths
