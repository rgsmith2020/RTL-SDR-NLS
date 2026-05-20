import ctypes
import os

print("Testing direct DLL load...")
try:
    dll_path = os.path.join(os.path.dirname(__file__), "rtlsdr.dll")
    print(f"Loading from: {dll_path}")
    lib = ctypes.CDLL(dll_path)
    print("rtlsdr.dll loaded successfully!")
    
    # Check if the function exists
    if hasattr(lib, "rtlsdr_set_dithering"):
        print("SUCCESS! rtlsdr_set_dithering is present in THIS DLL.")
    else:
        print("FAIL! rtlsdr_set_dithering is missing from THIS DLL.")
except Exception as e:
    print(f"Failed to load rtlsdr.dll: {e}")
