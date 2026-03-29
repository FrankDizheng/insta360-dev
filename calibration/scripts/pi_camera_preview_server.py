import io
import argparse
import os
import socketserver
from http import server

import cv2

from handeye_board_runtime import detect_camera_device


PORT = 8000
CAMERA_DEVICE = "auto"
WIDTH = 1280
HEIGHT = 720
JPEG_QUALITY = 85


PAGE = b"""\
<html>
  <head>
    <title>Gemini 335 Preview</title>
    <style>
      body { font-family: sans-serif; margin: 0; background: #111; color: #eee; }
      .wrap { padding: 16px; }
      img { max-width: 100%; height: auto; border: 1px solid #444; }
    </style>
  </head>
  <body>
    <div class="wrap">
      <h2>Gemini 335 Preview</h2>
      <p>Use this page while collecting hand-eye samples.</p>
      <img src="/stream.mjpg" />
    </div>
  </body>
</html>
"""


class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(301)
            self.send_header("Location", "/index.html")
            self.end_headers()
            return
        if self.path == "/index.html":
            content = PAGE
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
            return
        if self.path != "/stream.mjpg":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Age", 0)
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
        self.end_headers()

        try:
            while True:
                ok, frame = self.server.cap.read()
                if not ok or frame is None:
                    continue
                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
                )
                if not ok:
                    continue
                data = encoded.tobytes()
                self.wfile.write(b"--FRAME\r\n")
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", len(data))
                self.end_headers()
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
        except Exception:
            pass

    def log_message(self, format, *args):
        return


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, handler, cap):
        super().__init__(address, handler)
        self.cap = cap


def main():
    parser = argparse.ArgumentParser(description="Preview Orbbec color stream over MJPEG.")
    parser.add_argument("--camera-device", default=os.environ.get("CAMERA_DEVICE", CAMERA_DEVICE), help="V4L2 color device path or 'auto'")
    parser.add_argument("--width", type=int, default=WIDTH, help="Preview width")
    parser.add_argument("--height", type=int, default=HEIGHT, help="Preview height")
    parser.add_argument("--port", type=int, default=PORT, help="HTTP port")
    args = parser.parse_args()

    selected_device = detect_camera_device(args.camera_device)
    cap = cv2.VideoCapture(selected_device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {selected_device}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    with StreamingServer(("", args.port), StreamingHandler, cap) as httpd:
        print(f"preview_camera_device: {selected_device}", flush=True)
        print(f"preview_url: http://0.0.0.0:{args.port}/index.html", flush=True)
        try:
            httpd.serve_forever()
        finally:
            cap.release()


if __name__ == "__main__":
    main()
