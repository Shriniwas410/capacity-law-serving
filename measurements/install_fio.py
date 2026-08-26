#!/usr/bin/env python3
import tarfile, urllib.request, os, subprocess, pathlib
url = "https://github.com/axboe/fio/archive/refs/tags/fio-3.39.tar.gz"
dst = pathlib.Path("/tmp/fio-3.39.tar.gz")
print("downloading", url)
urllib.request.urlretrieve(url, dst)
print("extract")
with tarfile.open(dst, "r:gz") as t:
    t.extractall("/tmp")
os.chdir("/tmp/fio-fio-3.39")
subprocess.check_call(["./configure", f"--prefix={os.path.expanduser('~/.local')}"])
subprocess.check_call(["make", f"-j{os.cpu_count() or 4}"])
subprocess.check_call(["make", "install"])
print("installed")
