#!/usr/bin/env python3
"""
Simple RTSP server using FFmpeg.
This script uses FFmpeg to create an RTSP server for testing.
"""

import subprocess
import argparse
import sys
from pathlib import Path
import signal
import os


class SimpleRTSPServer:
    def __init__(self, port=8554):
        """
        Initialize RTSP server using FFmpeg
        
        Args:
            port: Port to run the RTSP server on
        """
        self.port = port
        self.process = None
        
    def stream_video_file(self, video_path, stream_name="test"):
        """
        Stream a video file via RTSP
        
        Args:
            video_path: Path to the video file
            stream_name: Name of the stream (rtsp://localhost:port/stream_name)
        """
        # FFmpeg command to stream video file via RTSP
        # This re-encodes the video and loops it
        cmd = [
            'ffmpeg',
            '-re',  # Read input at native frame rate
            '-stream_loop', '-1',  # Loop the video infinitely
            '-i', video_path,
            '-c:v', 'libx264',  # H.264 video codec
            '-preset', 'ultrafast',  # Fast encoding
            '-tune', 'zerolatency',  # Low latency
            '-b:v', '2000k',  # Video bitrate
            '-maxrate', '2000k',
            '-bufsize', '4000k',
            '-pix_fmt', 'yuv420p',
            '-g', '30',  # Keyframe interval
            '-f', 'rtsp',  # Output format
            f'rtsp://localhost:{self.port}/{stream_name}'
        ]
        
        print(f"\n{'='*60}")
        print(f"RTSP Server Starting")
        print(f"{'='*60}")
        print(f"Stream URL: rtsp://localhost:{self.port}/{stream_name}")
        print(f"Source: {video_path}")
        print(f"\nTest with VLC or ffplay:")
        print(f"  vlc rtsp://localhost:{self.port}/{stream_name}")
        print(f"  ffplay rtsp://localhost:{self.port}/{stream_name}")
        print(f"\nPress Ctrl+C to stop the server")
        print(f"{'='*60}\n")
        
        try:
            self.process = subprocess.Popen(cmd)
            self.process.wait()
        except KeyboardInterrupt:
            print("\n\nShutting down RTSP server...")
            if self.process:
                self.process.terminate()
                self.process.wait()
                
    def stream_webcam(self, device=0, stream_name="test"):
        """
        Stream webcam via RTSP
        
        Args:
            device: Webcam device number
            stream_name: Name of the stream
        """
        cmd = [
            'ffmpeg',
            '-f', 'v4l2',
            '-i', f'/dev/video{device}',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-b:v', '2000k',
            '-maxrate', '2000k',
            '-bufsize', '4000k',
            '-pix_fmt', 'yuv420p',
            '-g', '30',
            '-f', 'rtsp',
            f'rtsp://localhost:{self.port}/{stream_name}'
        ]
        
        print(f"\n{'='*60}")
        print(f"RTSP Server Starting")
        print(f"{'='*60}")
        print(f"Stream URL: rtsp://localhost:{self.port}/{stream_name}")
        print(f"Source: /dev/video{device}")
        print(f"\nTest with VLC or ffplay:")
        print(f"  vlc rtsp://localhost:{self.port}/{stream_name}")
        print(f"  ffplay rtsp://localhost:{self.port}/{stream_name}")
        print(f"\nPress Ctrl+C to stop the server")
        print(f"{'='*60}\n")
        
        try:
            self.process = subprocess.Popen(cmd)
            self.process.wait()
        except KeyboardInterrupt:
            print("\n\nShutting down RTSP server...")
            if self.process:
                self.process.terminate()
                self.process.wait()
                
    def stream_test_pattern(self, stream_name="test"):
        """
        Stream a test pattern via RTSP
        
        Args:
            stream_name: Name of the stream
        """
        cmd = [
            'ffmpeg',
            '-re',
            '-f', 'lavfi',
            '-i', 'testsrc=size=640x480:rate=30',  # Test pattern
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-b:v', '2000k',
            '-maxrate', '2000k',
            '-bufsize', '4000k',
            '-pix_fmt', 'yuv420p',
            '-g', '30',
            '-f', 'rtsp',
            f'rtsp://localhost:{self.port}/{stream_name}'
        ]
        
        print(f"\n{'='*60}")
        print(f"RTSP Server Starting")
        print(f"{'='*60}")
        print(f"Stream URL: rtsp://localhost:{self.port}/{stream_name}")
        print(f"Source: FFmpeg test pattern")
        print(f"\nTest with VLC or ffplay:")
        print(f"  vlc rtsp://localhost:{self.port}/{stream_name}")
        print(f"  ffplay rtsp://localhost:{self.port}/{stream_name}")
        print(f"\nPress Ctrl+C to stop the server")
        print(f"{'='*60}\n")
        
        try:
            self.process = subprocess.Popen(cmd)
            self.process.wait()
        except KeyboardInterrupt:
            print("\n\nShutting down RTSP server...")
            if self.process:
                self.process.terminate()
                self.process.wait()


def main():
    parser = argparse.ArgumentParser(
        description="Simple RTSP server using FFmpeg",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test pattern
  python simple_rtsp_server.py --test-pattern
  
  # Video file
  python simple_rtsp_server.py --video /path/to/video.mp4
  
  # Webcam
  python simple_rtsp_server.py --webcam 0
  
  # Custom port and stream name
  python simple_rtsp_server.py --test-pattern --port 8555 --stream mystream

Note: This requires FFmpeg to be installed and an RTSP server (like MediaMTX/rtsp-simple-server)
      to be running on the specified port to receive the stream.
      
Alternative: Use MediaMTX standalone (recommended):
  Download from: https://github.com/bluenviron/mediamtx/releases
  Run: ./mediamtx
  Then publish with: ffmpeg -re -i video.mp4 -c copy -f rtsp rtsp://localhost:8554/mystream
        """
    )
    
    # Source options (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--video",
        type=str,
        help="Path to video file to stream"
    )
    source_group.add_argument(
        "--webcam",
        type=int,
        metavar="DEVICE",
        help="Webcam device number (e.g., 0 for /dev/video0)"
    )
    source_group.add_argument(
        "--test-pattern",
        action="store_true",
        help="Use FFmpeg test pattern"
    )
    
    # Server options
    parser.add_argument(
        "--port",
        type=int,
        default=8554,
        help="RTSP server port (default: 8554)"
    )
    parser.add_argument(
        "--stream",
        type=str,
        default="test",
        help="Stream name/path (default: test)"
    )
    
    args = parser.parse_args()
    
    # Check if FFmpeg is installed
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: FFmpeg is not installed or not in PATH")
        print("Install with: sudo apt-get install ffmpeg")
        sys.exit(1)
    
    # Create RTSP server
    server = SimpleRTSPServer(port=args.port)
    
    # Stream appropriate source
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Error: Video file not found: {args.video}")
            sys.exit(1)
        server.stream_video_file(str(video_path.absolute()), args.stream)
    elif args.webcam is not None:
        webcam_device = f"/dev/video{args.webcam}"
        if not Path(webcam_device).exists():
            print(f"Error: Webcam device not found: {webcam_device}")
            sys.exit(1)
        server.stream_webcam(args.webcam, args.stream)
    elif args.test_pattern:
        server.stream_test_pattern(args.stream)


if __name__ == "__main__":
    main()
