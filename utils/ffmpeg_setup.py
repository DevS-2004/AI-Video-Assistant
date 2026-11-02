import os
import sys
import shutil


def ensure_ffmpeg():
    """Ensure ffmpeg and ffprobe are in os.environ['PATH'] and return ffmpeg path."""
    # 1. Already in PATH
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    # 2. Virtual environment Scripts / bin directory
    scripts_dir = os.path.dirname(sys.executable)
    exe_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    candidate = os.path.join(scripts_dir, exe_name)
    if os.path.exists(candidate):
        os.environ["PATH"] = scripts_dir + os.pathsep + os.environ.get("PATH", "")
        return candidate

    # 3. Project-local .venv\Scripts
    cwd_venv = os.path.join(os.getcwd(), ".venv", "Scripts" if sys.platform == "win32" else "bin")
    candidate = os.path.join(cwd_venv, exe_name)
    if os.path.exists(candidate):
        os.environ["PATH"] = cwd_venv + os.pathsep + os.environ.get("PATH", "")
        return candidate

    # 4. Check Windows WinGet location
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    winget_dir = os.path.join(local_app_data, "Microsoft", "WinGet", "Packages")
    if os.path.exists(winget_dir):
        for root, _, files in os.walk(winget_dir):
            if "ffmpeg.exe" in files:
                os.environ["PATH"] = root + os.pathsep + os.environ.get("PATH", "")
                return os.path.join(root, "ffmpeg.exe")

    return None


# Run automatically on import
FFMPEG_PATH = ensure_ffmpeg()
