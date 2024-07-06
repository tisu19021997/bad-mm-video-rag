from pathlib import Path

from preprocess import AppConfig, RunConfig, VideoIngestionPipeline

input_path_or_url = "https://www.youtube.com/watch?v=K2VOP6Cy4eI"

if __name__ == "__main__":
    run_config = RunConfig(seconds_per_frame=5)
    vid_pipe = VideoIngestionPipeline(AppConfig(save_dir=Path("my_dir")))
    vid_pipe.run(input_path_or_url, run_config)
