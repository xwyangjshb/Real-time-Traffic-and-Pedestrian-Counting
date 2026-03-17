"""
Flask 服务：提供视频统计 API 和前端页面。

依赖: pip install flask

启动: python server.py
访问: http://localhost:5000
"""

import os
import re
import json
import glob
from flask import Flask, jsonify, send_file, send_from_directory, abort, request, Response

OUTPUT_DIR = "./output"
WEB_DIR    = "./web"

app = Flask(__name__)


@app.route("/")
def index():
    return send_file(os.path.join(WEB_DIR, "index.html"))


@app.route("/api/video_list")
def video_list():
    """列出所有已处理视频的摘要（按处理时间倒序）。"""
    pattern = os.path.join(OUTPUT_DIR, "*_stats.json")
    videos  = []
    for path in sorted(glob.glob(pattern), reverse=True):
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            videos.append({
                "video_id":             data.get("video_id", ""),
                "processed_at":         data.get("processed_at", ""),
                "resolution":           data.get("resolution", ""),
                "duration_processed_s": data.get("duration_processed_s", 0),
                "frames_processed":     data.get("frames_processed", 0),
                "totals":               data.get("totals", {}),
                "annotated_video":      data.get("annotated_video", ""),
            })
        except Exception:
            pass
    return jsonify(videos)


@app.route("/api/video_stats/<video_id>")
def video_stats(video_id):
    """返回指定视频的完整统计 JSON（含逐帧数据）。"""
    path = os.path.join(OUTPUT_DIR, f"{video_id}_stats.json")
    if not os.path.exists(path):
        abort(404)
    with open(path, encoding='utf-8') as f:
        return jsonify(json.load(f))


@app.route("/output/<path:filename>")
def serve_output(filename):
    """供应 output 目录中的文件，mp4 支持 Range 请求（浏览器视频播放必须）。"""
    filepath = os.path.join(os.path.abspath(OUTPUT_DIR), filename)
    if not os.path.exists(filepath):
        abort(404)

    if not filename.lower().endswith('.mp4'):
        return send_from_directory(os.path.abspath(OUTPUT_DIR), filename)

    filesize    = os.path.getsize(filepath)
    range_header = request.headers.get('Range')

    if not range_header:
        resp = Response(open(filepath, 'rb').read(), 200, mimetype='video/mp4')
        resp.headers['Accept-Ranges'] = 'bytes'
        resp.headers['Content-Length'] = filesize
        return resp

    m = re.search(r'bytes=(\d+)-(\d*)', range_header)
    byte1 = int(m.group(1))
    byte2 = int(m.group(2)) if m.group(2) else filesize - 1
    length = byte2 - byte1 + 1

    with open(filepath, 'rb') as f:
        f.seek(byte1)
        data = f.read(length)

    resp = Response(data, 206, mimetype='video/mp4')
    resp.headers['Content-Range']  = f'bytes {byte1}-{byte2}/{filesize}'
    resp.headers['Accept-Ranges']  = 'bytes'
    resp.headers['Content-Length'] = str(length)
    return resp


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(WEB_DIR, exist_ok=True)
    print("[服务启动] http://localhost:8080")
    app.run(host="0.0.0.0", port=8080, debug=False)
