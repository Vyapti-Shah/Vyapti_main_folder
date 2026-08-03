# service/pipeline_service.py
import matplotlib.pyplot as plt
from .video_service import VideoService
from .filter_service import FilterService
import config

class PipelineService:
    def __init__(self):
        self.video_service = VideoService()
        self.filter_service = FilterService()

    def process(self, input_path, output_path):
        cap, width, height, fps, total_frames = self.video_service.open_video(input_path)
        writer = self.video_service.create_writer(output_path, fps, width, height)
        
        print(f"[INFO] Processing {total_frames} frames via Pipeline Service...")
        
        try:
            import tqdm
            pbar = tqdm.tqdm(total=total_frames)
            use_tqdm = True
        except ImportError:
            use_tqdm = False

        frame_count = 0
        input_noise_levels = []
        output_noise_levels = []

        while True:
            ret, frame = self.video_service.read_frame(cap)
            if not ret:
                break
                
            # Measure noise of original raw frame
            input_noise = self.filter_service.calculate_noise(frame)
            input_noise_levels.append(input_noise)

            # Denoise & Sharpen
            denoised_frame = self.filter_service.denoise(frame)
            final_frame = self.filter_service.sharpen(denoised_frame)
            
            # Measure noise of processed frame
            output_noise = self.filter_service.calculate_noise(final_frame)
            output_noise_levels.append(output_noise)
            
            self.video_service.write_frame(writer, final_frame)
            
            frame_count += 1
            if use_tqdm:
                pbar.update(1)
            elif frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")

        if use_tqdm:
            pbar.close()

        self.cleanup(cap, writer)
        print(f"\n[SUCCESS] Saved highly smoothed video to {output_path}")
        
        # Plot noise reduction
        self.generate_comparison_graph(input_noise_levels, output_noise_levels)
        
        # Calculate and log metrics
        self.log_metrics(input_noise_levels, output_noise_levels, total_frames)

    def cleanup(self, cap, writer):
        cap.release()
        writer.release()
        
    def log_metrics(self, input_noise, output_noise, total_frames):
        avg_input = sum(input_noise) / len(input_noise) if input_noise else 0
        avg_output = sum(output_noise) / len(output_noise) if output_noise else 0
        reduction_percent = ((avg_input - avg_output) / avg_input) * 100 if avg_input > 0 else 0
        
        print("\n--- DENOISING METRICS SUMMARY ---")
        print(f"Total Frames Processed: {total_frames}")
        print(f"Average Input Noise (Variance):  {avg_input:.2f}")
        print(f"Average Output Noise (Variance): {avg_output:.2f}")
        print(f"Total Noise Reduction:           {reduction_percent:.2f}%")
        print("---------------------------------")
        
        # Save frame-by-frame metrics to a CSV log
        log_file = "denoising_metrics.csv"
        try:
            with open(log_file, 'w') as f:
                f.write("Frame,InputVariance,OutputVariance,ReductionPercent\n")
                for i in range(len(input_noise)):
                    inp = input_noise[i]
                    out = output_noise[i]
                    red = ((inp - out) / inp * 100) if inp > 0 else 0
                    f.write(f"{i+1},{inp:.2f},{out:.2f},{red:.2f}\n")
            print(f"[SUCCESS] Saved detailed frame-by-frame metrics to {log_file}")
        except Exception as e:
            print(f"[ERROR] Could not write metrics log: {e}")
            
    def generate_comparison_graph(self, input_noise, output_noise):
        print("[INFO] Generating Denoise Comparison Graph...")
        plt.figure(figsize=(10, 5))
        plt.plot(input_noise, label='Raw Input Noise (Laplacian Variance)', color='red', alpha=0.7)
        plt.plot(output_noise, label='Smoothed Output Noise', color='blue', alpha=0.9)
        plt.title("Noise / Texture Variance Per Frame")
        plt.xlabel("Frame Number")
        plt.ylabel("Variance (Texture Strength)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(config.PLOT_FILENAME)
        plt.close()
        print(f"[SUCCESS] Saved graph to {config.PLOT_FILENAME}")
