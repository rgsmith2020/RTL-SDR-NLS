import os

rtlsdr_path = r"C:\Users\Rsmit\OneDrive\Desktop\_Python_\RTL-SDR-NLS\.venv\Lib\site-packages\rtlsdr\rtlsdr.py"

with open(rtlsdr_path, 'r') as f:
    content = f.read()

# The python class itself is explicitly calling librtlsdr.rtlsdr_set_dithering
# We need to wrap that call inside the Python class.

target = """        # disable PLL dithering if necessary. If it's going to happen, it must
        # happen before frequency is set.
        result = librtlsdr.rtlsdr_set_dithering(self.dev_p, int(dithering_enabled))
        if result < 0:
            raise IOError('Error code %d when setting PLL dithering mode'\\
                           % (result))"""

replacement = """        # disable PLL dithering if necessary. If it's going to happen, it must
        # happen before frequency is set.
        try:
            result = librtlsdr.rtlsdr_set_dithering(self.dev_p, int(dithering_enabled))
            if result < 0:
                raise IOError('Error code %d when setting PLL dithering mode'\\
                               % (result))
        except AttributeError:
            pass # Ignore missing dithering function"""

content = content.replace(target, replacement)

with open(rtlsdr_path, 'w') as f:
    f.write(content)

print("Patched rtlsdr.py (class file) successfully!")
