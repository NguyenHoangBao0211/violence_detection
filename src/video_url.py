"""Download direct public HTTP(S) video links with bounded size and redirects."""
import http.client
import importlib.util
import json
import subprocess
import sys
import tempfile
import shutil
import ipaddress
import socket
import ssl
import time
from pathlib import Path
from urllib.parse import urlsplit, urljoin, unquote

MAX_BYTES = 200 * 1024 * 1024

def _download_direct(url, destination, headers=None):
    deadline = time.monotonic() + 120
    for _ in range(6):
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
            raise ValueError("Chỉ hỗ trợ link video HTTP/HTTPS công khai.")
        port = parts.port or (443 if parts.scheme == "https" else 80)
        if port not in {80, 443}:
            raise ValueError("Link phải dùng cổng HTTP/HTTPS tiêu chuẩn.")
        addresses = socket.getaddrinfo(parts.hostname, port, type=socket.SOCK_STREAM)
        if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
            raise ValueError("Không hỗ trợ địa chỉ nội bộ hoặc địa chỉ riêng.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("Quá thời gian tải video.")
        conn = http.client.HTTPConnection(parts.hostname, port, timeout=min(20, remaining))
        try:
            # Connect to the validated address; TLS still verifies the URL hostname.
            conn.sock = socket.create_connection(addresses[0][4][:2], timeout=min(20, remaining))
            if parts.scheme == "https":
                conn.sock = ssl.create_default_context().wrap_socket(conn.sock, server_hostname=parts.hostname)
            target = parts.path or "/"
            if parts.query:
                target += "?" + parts.query
            conn.request("GET", target, headers={"User-Agent": "ViolenceDetection/2.1", "Accept": "video/*, application/octet-stream", **(headers or {})})
            response = conn.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader("Location")
                if not location:
                    raise ValueError("Link chuyển hướng không hợp lệ.")
                url = urljoin(url, location)
                continue
            if response.status != 200:
                raise ValueError(f"Không tải được video (HTTP {response.status}).")
            content_type = response.getheader("Content-Type", "").split(";")[0].lower()
            if content_type.startswith("text/") or content_type in {"application/json", "application/xhtml+xml"}:
                raise ValueError("Link trả về trang web. Hãy dùng link tải trực tiếp video.")
            length = response.getheader("Content-Length")
            if length and int(length) > MAX_BYTES:
                raise ValueError("Video vượt quá giới hạn 200 MB.")
            total = 0
            with Path(destination).open("wb") as output:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ValueError("Quá thời gian tải video.")
                    if conn.sock:
                        conn.sock.settimeout(min(20, remaining))
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_BYTES:
                        raise ValueError("Video vượt quá giới hạn 200 MB.")
                    output.write(chunk)
            if not total:
                raise ValueError("Video trống.")
            name = Path(unquote(parts.path)).name or "video.mp4"
            return name, total
        finally:
            conn.close()
    raise ValueError("Link chuyển hướng quá nhiều lần.")


def _social_platform(url):
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    if host == "tiktok.com" or host.endswith(".tiktok.com"):
        return "TikTok"
    if host == "facebook.com" or host.endswith(".facebook.com") or host in {"fb.watch", "www.fb.watch"}:
        return "Facebook"
    return None


def download_video(url, destination):
    platform = _social_platform(url)
    if not platform:
        return _download_direct(url, destination)
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or parts.username or parts.password or parts.port not in {None, 80, 443}:
        raise ValueError("Link TikTok/Facebook không hợp lệ.")
    if importlib.util.find_spec("yt_dlp") is None:
        raise ValueError("Chưa cài thư viện lấy video. Chạy: python -m pip install -U yt-dlp, rồi khởi động lại backend.")
    try:
        # Extract metadata only; download the chosen media via the validated downloader.
        result = subprocess.run([
            sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-playlist",
            "--no-warnings", "--skip-download", "--dump-single-json",
            "--use-extractors", "TikTok.*,Facebook.*", "--socket-timeout", "15",
            "--retries", "1", "--extractor-retries", "1", "--format",
            "best[ext=mp4][protocol=https]/best[ext=mp4][protocol=http]/best[protocol=https]/best[protocol=http]",
            "--", url,
        ], capture_output=True, text=True, encoding="utf-8", timeout=90,
           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"Quá thời gian lấy video {platform}. Hãy thử lại.") from exc
    if result.returncode:
        raise ValueError(f"Không lấy được video {platform}. Video có thể bị giới hạn, yêu cầu đăng nhập hoặc nền tảng đang chặn tải. Hãy thử link video công khai khác.")
    try:
        info = json.loads(result.stdout)
    except (ValueError, TypeError) as exc:
        raise ValueError("Không đọc được thông tin video.") from exc
    if info.get("is_live") or info.get("_type") in {"playlist", "multi_video"}:
        raise ValueError("Chỉ hỗ trợ một video đã đăng, không hỗ trợ livestream hoặc danh sách video.")
    media_url = info.get("url")
    if not media_url:
        raise ValueError("Không tìm thấy link tải video trực tiếp.")
    if (info.get("filesize") or 0) > MAX_BYTES:
        raise ValueError("Video vượt quá giới hạn 200 MB.")
    headers = {k: v for k, v in (info.get("http_headers") or {}).items()
               if k.lower() in {"user-agent", "referer", "accept"}}
    try:
        _, size = _download_direct(media_url, destination, headers=headers)
    except ValueError as exc:
        if "HTTP 403" not in str(exc):
            raise
        # Some social CDNs require the extractor session and browser impersonation.
        size = _download_social_session(url, destination)
    video_id = "".join(ch for ch in str(info.get("id", "video")) if ch.isalnum() or ch in "-_")[:80]
    extension = info.get("ext", "mp4")
    if extension not in {"mp4", "webm", "mov", "mkv", "avi"}:
        extension = "mp4"
    return f"{platform.lower()}_{video_id}.{extension}", size


def _download_social_session(url, destination):
    with tempfile.TemporaryDirectory(prefix="vd-social-") as directory:
        output = Path(directory) / "video.mp4"
        with (Path(directory) / "error.log").open("wb") as errors:
            process = subprocess.Popen([
                sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-playlist",
                "--use-extractors", "TikTok.*,Facebook.*", "--no-warnings", "--no-progress",
                "--socket-timeout", "15", "--retries", "1", "--extractor-retries", "1",
                "--max-filesize", str(MAX_BYTES), "--no-part", "--format",
                "best[ext=mp4][protocol=https]/best[ext=mp4][protocol=http]",
                "--output", str(output), "--", url,
            ], stdout=subprocess.DEVNULL, stderr=errors,
               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            deadline = time.monotonic() + 120
            try:
                while process.poll() is None:
                    if output.exists() and output.stat().st_size > MAX_BYTES:
                        raise ValueError("Video vượt quá giới hạn 200 MB.")
                    if time.monotonic() > deadline:
                        raise ValueError("Quá thời gian tải video từ mạng xã hội.")
                    time.sleep(0.2)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
        if process.returncode or not output.exists() or not output.stat().st_size:
            raise ValueError("Nền tảng từ chối tải video. Hãy thử link công khai khác; video yêu cầu đăng nhập chưa được hỗ trợ.")
        size = output.stat().st_size
        if size > MAX_BYTES:
            raise ValueError("Video vượt quá giới hạn 200 MB.")
        shutil.copyfile(output, destination)
        return size
