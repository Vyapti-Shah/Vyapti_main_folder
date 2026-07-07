import argparse
from controller.denoise_controller import DenoiseController

def main():
    parser = argparse.ArgumentParser(description="Pure Python MVC Denoising Pipeline")
    parser.add_argument("-i", "--input", required=True, help="Path to input video")
    parser.add_argument("-o", "--output", required=True, help="Path to output video")
    
    args = parser.parse_args()
    
    controller = DenoiseController()
    controller.run(args.input, args.output)

if __name__ == "__main__":
    main()
