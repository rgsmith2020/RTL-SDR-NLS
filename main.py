import os
import sys

# Add the current directory to the DLL search path
if sys.platform == 'win32':
    os.environ['PATH'] = os.path.abspath(os.path.dirname(__file__)) + os.pathsep + os.environ.get('PATH', '')
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(os.path.abspath(os.path.dirname(__file__)))

from rtlsdr import RtlSdr
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QLabel, QPushButton, QProgressBar, QGridLayout)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont, QColor

# --- SDR CONFIGURATION ---
SAMPLE_RATE = 2.048e6  # 2.048 MHz
CENTER_FREQ = 100.0e6  # 100.0 MHz
FFT_SIZE = 1024
WATERFALL_HISTORY = 200

def get_waterfall_bins(sdr_device, size):
    """ Reads I/Q data from the SDR and calculates the PSD in dB. """
    samples = sdr_device.read_samples(size)
    if len(samples) != size:
        return None

    window = np.blackman(len(samples))
    windowed_data = samples * window
    fft_data = np.fft.fft(windowed_data)
    fft_shifted = np.fft.fftshift(fft_data)
    
    magnitude = np.abs(fft_shifted)
    magnitude = np.maximum(magnitude, 1e-9)
    power_db = 20 * np.log10(magnitude)
    return power_db

class SDRWindow(QMainWindow):
    def __init__(self, sdr_device):
        super().__init__()
        self.sdr = sdr_device
        self.setWindowTitle("IC-705 Style SDR")
        self.resize(1024, 768)
        
        # Apply dark hardware theme
        self.setStyleSheet("""
            QMainWindow { background-color: #0b0b0d; }
            QWidget { color: #ffffff; }
            QPushButton { 
                background-color: #1e1e24; 
                border: 2px solid #333; 
                border-radius: 5px; 
                font-weight: bold; 
                padding: 10px;
            }
            QPushButton:pressed { background-color: #333; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # --- 1. TOP HARDWARE PANEL ---
        top_panel = QWidget()
        top_layout = QGridLayout(top_panel)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        # VFO/Mode indicators
        self.vfo_label = QLabel("VFO A")
        self.vfo_label.setStyleSheet("color: #00ff00; font-size: 20px; font-weight: bold;")
        self.mode_label = QLabel("FM")
        self.mode_label.setStyleSheet("background-color: #2b2b36; border: 1px solid #555; padding: 5px; font-size: 24px; font-weight: bold;")
        self.filter_label = QLabel("FIL1")
        self.filter_label.setStyleSheet("color: #aaaaaa; font-size: 20px;")
        
        # Big Frequency Display
        self.freq_label = QLabel(f"{CENTER_FREQ / 1e6:.3f}.00")
        self.freq_label.setFont(QFont("Consolas", 64, QFont.Weight.Bold))
        # IC-705 often uses a very bright cyan/white for the main VFO
        self.freq_label.setStyleSheet("color: #c8f4ff; letter-spacing: 2px;")
        self.freq_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        top_layout.addWidget(self.vfo_label, 0, 0, Qt.AlignmentFlag.AlignTop)
        top_layout.addWidget(self.freq_label, 0, 1, 2, 1, Qt.AlignmentFlag.AlignCenter)
        top_layout.addWidget(self.mode_label, 0, 2, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        top_layout.addWidget(self.filter_label, 1, 2, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)

        layout.addWidget(top_panel)

        # --- 2. S-METER ---
        meter_layout = QHBoxLayout()
        meter_title = QLabel("S")
        meter_title.setStyleSheet("color: #ff3333; font-weight: bold; font-size: 18px;")
        
        self.meter = QProgressBar()
        self.meter.setTextVisible(False)
        self.meter.setFixedHeight(12)
        self.meter.setRange(-90, 0)  # -90dB to 0dB scale
        self.meter.setStyleSheet("""
            QProgressBar { border: 1px solid #444; background-color: #111; border-radius: 3px; }
            QProgressBar::chunk { background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0055ff, stop:0.7 #00ff00, stop:0.9 #ffff00, stop:1 #ff0000); }
        """)
        
        meter_layout.addWidget(meter_title)
        meter_layout.addWidget(self.meter)
        layout.addLayout(meter_layout)

        # --- 3. SPECTRUM & WATERFALL ---
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.ci.layout.setSpacing(0) # Merge the spectrum and waterfall seamlessly
        layout.addWidget(self.glw, stretch=1)

        # Spectrum Scope (Filled)
        self.spectrum_plot = self.glw.addPlot(row=0, col=0)
        self.spectrum_plot.hideAxis('left')
        self.spectrum_plot.hideAxis('bottom')
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.2)
        self.spectrum_plot.setYRange(-90, -10)
        
        # IC-705 uses a semi-transparent fill for the scope
        pen = pg.mkPen(color='#88ffff', width=1.5)
        brush = pg.mkBrush(color=(0, 255, 255, 60))
        self.spectrum_curve = self.spectrum_plot.plot(fillLevel=-100, brush=brush, pen=pen)

        # Waterfall (Falls DOWN from the spectrum)
        self.waterfall_plot = self.glw.addPlot(row=1, col=0)
        self.waterfall_plot.hideAxis('left')
        self.waterfall_plot.setLabel('bottom', "MHz")
        self.waterfall_plot.getAxis('bottom').setPen('#fff')
        self.waterfall_plot.getAxis('bottom').setTextPen('#fff')
        
        self.waterfall_image = pg.ImageItem()
        self.waterfall_plot.addItem(self.waterfall_image)
        
        # IC-705 colormap: dark blue -> lighter blue -> yellow -> red -> white
        pos = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
        color = np.array([[0, 0, 50, 255], [0, 50, 200, 255], [200, 200, 0, 255], [255, 50, 0, 255], [255, 255, 255, 255]], dtype=np.ubyte)
        colormap = pg.ColorMap(pos, color)
        self.waterfall_image.setColorMap(colormap)
        
        # Shape: (X=Frequency, Y=Time). We want Y=max to be at the top.
        self.waterfall_data = np.full((FFT_SIZE, WATERFALL_HISTORY), -100.0)
        self.waterfall_image.setLevels([-85, -20]) 

        # Calculate frequency X-axis bounds
        freq_axis = np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, 1/SAMPLE_RATE))
        self.freqs_mhz = (CENTER_FREQ + freq_axis) / 1e6
        x_start, x_span = self.freqs_mhz[0], self.freqs_mhz[-1] - self.freqs_mhz[0]
        
        self.waterfall_image.setRect(pg.QtCore.QRectF(x_start, 0, x_span, WATERFALL_HISTORY))
        self.spectrum_plot.setXRange(x_start, x_start + x_span, padding=0)
        self.waterfall_plot.setXRange(x_start, x_start + x_span, padding=0)
        self.waterfall_plot.setXLink(self.spectrum_plot)

        # --- 4. BOTTOM TOUCHSCREEN BUTTONS ---
        btn_layout = QHBoxLayout()
        for text in ["MENU", "F-INP", "VFO/MEM", "A/B", "MW", "QUICK"]:
            btn = QPushButton(text)
            btn.setFixedHeight(50)
            btn_layout.addWidget(btn)
        layout.addLayout(btn_layout)

        # --- Timer ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(50)

    def update_plots(self):
        power_spectrum = get_waterfall_bins(self.sdr, FFT_SIZE)
        if power_spectrum is None:
            return

        # 1. Update Spectrum Analyzer
        self.spectrum_curve.setData(self.freqs_mhz, power_spectrum)

        # 2. Update Waterfall image
        # Roll data down along the Y-axis (axis=1) so older data moves to Y=0 (bottom of screen)
        self.waterfall_data = np.roll(self.waterfall_data, -1, axis=1)
        # Put the newest data at the top row (Y = max-1)
        self.waterfall_data[:, -1] = power_spectrum
        self.waterfall_image.setImage(self.waterfall_data, autoLevels=False)

        # 3. Update S-Meter (Measure center frequency power)
        # We take a small slice around the center bin to simulate the active receiver bandwidth
        center_bin = FFT_SIZE // 2
        signal_power = np.max(power_spectrum[center_bin-5 : center_bin+5])
        self.meter.setValue(int(signal_power))

    def closeEvent(self, event):
        self.timer.stop()
        self.sdr.close()
        event.accept()

if __name__ == '__main__':
    sdr = RtlSdr()
    sdr.sample_rate = SAMPLE_RATE
    sdr.center_freq = CENTER_FREQ
    sdr.gain = 'auto'

    app = QApplication(sys.argv)
    window = SDRWindow(sdr)
    window.show()
    sys.exit(app.exec())
