import urllib.request
import zipfile
import os
import shutil

# PothosSDR provides extremely stable Windows builds that include ALL symbols
url = "https://downloads.myriadrf.org/builds/PothosSDR/PothosSDR-2021.07.25-vc16-x64.exe"
# Wait, downloading an exe is hard to extract in python without 7z.
# Let's try to patch the pyrtlsdr python code directly to ignore the missing function if it's not strictly necessary for basic tuning!
