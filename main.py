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
import sounddevice as sd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QLabel, QPushButton, QSlider, QCheckBox, 
                             QComboBox, QGroupBox, QDoubleSpinBox)
from PyQt6.QtCore import QTimer, Qt

# --- CONSTANTS ---
FFT_SIZE = 2048
WATERFALL_HISTORY = 300
AUDIO_RATE = 48000
AUDIO_CHUNK = 65536  # IQ samples per frame when FM is active (~32 ms at 2.048 MSPS)

def get_waterfall_bins(samples):
    """ Calculates PSD in dB from an IQ sample array of length FFT_SIZE. """
    win = np.blackman(len(samples))
    fft_shifted = np.fft.fftshift(np.fft.fft(samples * win))
    return 20 * np.log10(np.maximum(np.abs(fft_shifted), 1e-9))

class SDRWindow(QMainWindow):
    def __init__(self, sdr_device):
        super().__init__()
        self.sdr = sdr_device
        self.playing = True
        self.avg_spectrum = None
        self.audio_stream = None
        self.deemph_state = 0.0
        self.sample_rate = 2.048e6
        self.setWindowTitle("SDRSharp Style Tuner")
        self.resize(1100, 800)
        
        self.setStyleSheet("""
            QMainWindow, QWidget          { background-color: #f0f0f0; color: #1a1a1a; }
            QGroupBox                     { font-weight: bold; border: 1px solid #bbb;
                                            border-radius: 5px; margin-top: 10px; color: #1a1a1a; }
            QGroupBox::title              { subcontrol-origin: margin; left: 10px;
                                            padding: 0 3px 0 3px; color: #1a1a1a; }
            QLabel                        { color: #1a1a1a; }
            QCheckBox                     { color: #1a1a1a; }
            QPushButton                   { color: #1a1a1a; background-color: #e0e0e0;
                                            border: 1px solid #bbb; border-radius: 3px;
                                            padding: 3px 8px; }
            QPushButton:hover             { background-color: #d4d4d4; }
            QPushButton:pressed           { background-color: #c0c0c0; }
            QComboBox                     { color: #1a1a1a; background-color: #ffffff;
                                            border: 1px solid #bbb; border-radius: 3px;
                                            padding: 2px 4px; }
            QComboBox QAbstractItemView   { color: #1a1a1a; background-color: #ffffff;
                                            selection-background-color: #0078d7;
                                            selection-color: #ffffff; }
            QDoubleSpinBox                { color: #1a1a1a; background-color: #ffffff;
                                            border: 1px solid #bbb; border-radius: 3px;
                                            padding: 2px 4px; }
            QSlider::groove:horizontal    { background: #d0d0d0; height: 6px;
                                            border-radius: 3px; }
            QSlider::handle:horizontal    { background: #777; width: 14px; height: 14px;
                                            margin: -4px 0; border-radius: 7px; }
            QSlider::handle:horizontal:hover { background: #555; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- 1. LEFT CONTROL PANEL ---
        left_panel = QWidget()
        left_panel.setFixedWidth(250)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Play/Stop Button
        self.play_btn = QPushButton("Stop")
        self.play_btn.setMinimumHeight(40)
        self.play_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.play_btn.clicked.connect(self.toggle_play)
        left_layout.addWidget(self.play_btn)

        # -- Radio Settings Group --
        radio_group = QGroupBox("Radio")
        radio_layout = QVBoxLayout()

        radio_layout.addWidget(QLabel("Frequency (MHz):"))
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setDecimals(4)
        self.freq_spin.setRange(24.0, 1766.0) # Standard RTL-SDR range
        self.freq_spin.setSingleStep(0.1)
        self.freq_spin.setValue(100.0)
        self.freq_spin.valueChanged.connect(self.update_sdr_settings)
        radio_layout.addWidget(self.freq_spin)

        step_hlay = QHBoxLayout()
        self.step_down_btn = QPushButton("◀")
        self.step_down_btn.setFixedWidth(35)
        self.step_down_btn.clicked.connect(lambda: self._step_freq(-1))
        self.step_combo = QComboBox()
        self.step_combo.addItems(["1 kHz", "5 kHz", "10 kHz", "25 kHz", "100 kHz", "1 MHz"])
        self.step_combo.setCurrentIndex(2)
        self.step_up_btn = QPushButton("▶")
        self.step_up_btn.setFixedWidth(35)
        self.step_up_btn.clicked.connect(lambda: self._step_freq(1))
        step_hlay.addWidget(self.step_down_btn)
        step_hlay.addWidget(self.step_combo)
        step_hlay.addWidget(self.step_up_btn)
        radio_layout.addLayout(step_hlay)

        radio_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["WFM", "NFM", "AM", "USB", "LSB", "CW"])
        radio_layout.addWidget(self.mode_combo)

        radio_layout.addWidget(QLabel("Sample Rate:"))
        self.sr_combo = QComboBox()
        self.sr_combo.addItems(["1.024 MSPS", "2.048 MSPS", "2.400 MSPS", "3.200 MSPS"])
        self.sr_combo.setCurrentIndex(1) # Default 2.048
        self.sr_combo.currentIndexChanged.connect(self.update_sdr_settings)
        radio_layout.addWidget(self.sr_combo)

        self.gain_label = QLabel("RF Gain: Auto")
        radio_layout.addWidget(self.gain_label)

        gain_hlay = QHBoxLayout()
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(0, 49) # Approx max RTL-SDR gain
        self.gain_slider.setValue(20)
        self.gain_slider.setEnabled(False) # Disabled by default because of Auto Gain
        self.gain_slider.valueChanged.connect(self.update_sdr_settings)
        
        self.auto_gain_cb = QCheckBox("Auto")
        self.auto_gain_cb.setChecked(True)
        self.auto_gain_cb.toggled.connect(self.toggle_auto_gain)
        
        gain_hlay.addWidget(self.gain_slider)
        gain_hlay.addWidget(self.auto_gain_cb)
        radio_layout.addLayout(gain_hlay)
        radio_group.setLayout(radio_layout)
        left_layout.addWidget(radio_group)

        # -- FFT Display Group --
        fft_group = QGroupBox("FFT Display")
        fft_layout = QVBoxLayout()

        self.min_lvl_label = QLabel("Min Level: -85 dB")
        fft_layout.addWidget(self.min_lvl_label)
        self.min_lvl_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_lvl_slider.setRange(-120, 0)
        self.min_lvl_slider.setValue(-85)
        self.min_lvl_slider.valueChanged.connect(self.update_waterfall_levels)
        fft_layout.addWidget(self.min_lvl_slider)

        self.max_lvl_label = QLabel("Max Level: -20 dB")
        fft_layout.addWidget(self.max_lvl_label)
        self.max_lvl_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_lvl_slider.setRange(-120, 0)
        self.max_lvl_slider.setValue(-20)
        self.max_lvl_slider.valueChanged.connect(self.update_waterfall_levels)
        fft_layout.addWidget(self.max_lvl_slider)

        self.avg_label = QLabel("Averaging: 1")
        fft_layout.addWidget(self.avg_label)
        self.avg_slider = QSlider(Qt.Orientation.Horizontal)
        self.avg_slider.setRange(1, 20)
        self.avg_slider.setValue(1)
        self.avg_slider.valueChanged.connect(self.on_avg_changed)
        fft_layout.addWidget(self.avg_slider)

        fft_group.setLayout(fft_layout)
        left_layout.addWidget(fft_group)

        # -- Audio Group --
        audio_group = QGroupBox("Audio")
        audio_layout = QVBoxLayout()

        self.fm_cb = QCheckBox("Enable Audio")
        self.fm_cb.toggled.connect(self._toggle_fm)
        audio_layout.addWidget(self.fm_cb)

        self.vol_label = QLabel("Volume: 80%")
        audio_layout.addWidget(self.vol_label)
        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(80)
        self.vol_slider.valueChanged.connect(lambda v: self.vol_label.setText(f"Volume: {v}%"))
        audio_layout.addWidget(self.vol_slider)

        audio_group.setLayout(audio_layout)
        left_layout.addWidget(audio_group)

        main_layout.addWidget(left_panel)

        # --- 2. RIGHT PANEL (PLOTS) ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.ci.layout.setSpacing(0)
        right_layout.addWidget(self.glw)
        main_layout.addWidget(right_panel)

        # Spectrum Scope
        self.spectrum_plot = self.glw.addPlot(row=0, col=0)
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.3)
        self.spectrum_plot.setYRange(-100, 0)
        self.spectrum_plot.setLabel('left', "Power", units='dB')
        self.spectrum_curve = self.spectrum_plot.plot(pen=pg.mkPen('y', width=1.5), fillLevel=-120, brush=(255, 255, 0, 50))

        # Waterfall
        self.waterfall_plot = self.glw.addPlot(row=1, col=0)
        self.waterfall_plot.setLabel('bottom', "Frequency", units='MHz')
        
        self.waterfall_image = pg.ImageItem()
        self.waterfall_plot.addItem(self.waterfall_image)
        
        # Colormap similar to SDRSharp (Dark blue -> Red -> Yellow -> White)
        pos = np.array([0.0, 0.3, 0.6, 0.8, 1.0])
        color = np.array([[0, 0, 50, 255], [0, 50, 200, 255], [200, 0, 0, 255], [255, 200, 0, 255], [255, 255, 255, 255]], dtype=np.ubyte)
        self.waterfall_image.setColorMap(pg.ColorMap(pos, color))
        
        self.waterfall_data = np.full((FFT_SIZE, WATERFALL_HISTORY), -120.0)
        
        # Link X axes
        self.waterfall_plot.setXLink(self.spectrum_plot)

        # VFO marker and bandwidth overlay on spectrum
        self.vfo_line = pg.InfiniteLine(angle=90, pen=pg.mkPen('c', width=1))
        self.bw_region = pg.LinearRegionItem(movable=False, brush=pg.mkBrush(0, 200, 255, 40))
        self.spectrum_plot.addItem(self.bw_region)
        self.spectrum_plot.addItem(self.vfo_line)

        # Same overlay on waterfall
        self.wf_vfo_line = pg.InfiniteLine(angle=90, pen=pg.mkPen('c', width=1))
        self.wf_bw_region = pg.LinearRegionItem(movable=False, brush=pg.mkBrush(0, 200, 255, 25))
        self.waterfall_plot.addItem(self.wf_bw_region)
        self.waterfall_plot.addItem(self.wf_vfo_line)

        # Connect mode combo now that overlay items exist
        self.mode_combo.currentTextChanged.connect(self._update_bw_display)

        # Click-to-tune on spectrum and waterfall
        self.glw.scene().sigMouseClicked.connect(self._on_plot_click)

        # Apply initial settings to hardware and update plots
        self.update_sdr_settings()
        self.update_waterfall_levels()

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(40) # ~25 fps

    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("Stop" if self.playing else "Play")

    def toggle_auto_gain(self, checked):
        self.gain_slider.setEnabled(not checked)
        self.update_sdr_settings()

    def update_sdr_settings(self):
        """ Applies frequency, sample rate, and gain changes to the SDR hardware. """
        freq_hz = self.freq_spin.value() * 1e6
        sr_text = self.sr_combo.currentText().split()[0]
        sr_hz = float(sr_text) * 1e6
        self.sample_rate = sr_hz

        # Update Hardware
        try:
            self.sdr.sample_rate = sr_hz
            self.sdr.center_freq = freq_hz
            if self.auto_gain_cb.isChecked():
                self.sdr.gain = 'auto'
                self.gain_label.setText("RF Gain: Auto")
            else:
                self.sdr.gain = self.gain_slider.value()
                self.gain_label.setText(f"RF Gain: {self.sdr.gain} dB")
        except Exception as e:
            print(f"Hardware setting error: {e}")

        # Recalculate X-Axis ranges for the plots
        freq_axis = np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, 1/sr_hz))
        self.freqs_mhz = (freq_hz + freq_axis) / 1e6
        
        x_start = self.freqs_mhz[0]
        x_span = self.freqs_mhz[-1] - self.freqs_mhz[0]
        self.waterfall_image.setRect(pg.QtCore.QRectF(x_start, 0, x_span, WATERFALL_HISTORY))
        self.spectrum_plot.setXRange(x_start, x_start + x_span, padding=0)
        
        # Clear the waterfall history and averaging state on frequency/rate change
        self.waterfall_data.fill(-120.0)
        self.avg_spectrum = None
        self._update_bw_display()

    def on_avg_changed(self, value):
        self.avg_label.setText(f"Averaging: {value}")
        self.avg_spectrum = None

    def _on_plot_click(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.scenePos()
        if self.spectrum_plot.sceneBoundingRect().contains(pos):
            vb = self.spectrum_plot.getViewBox()
        elif self.waterfall_plot.sceneBoundingRect().contains(pos):
            vb = self.waterfall_plot.getViewBox()
        else:
            return
        freq_mhz = vb.mapSceneToView(pos).x()
        self.freq_spin.setValue(round(max(24.0, min(1766.0, freq_mhz)), 4))

    def _update_bw_display(self):
        if not hasattr(self, 'bw_region'):
            return
        center = self.freq_spin.value()
        mode = self.mode_combo.currentText()
        half_bw = {'WFM': 0.1, 'NFM': 0.00625, 'AM': 0.005, 'CW': 0.0005}
        if mode == 'USB':
            lo, hi = center + 0.0003, center + 0.003
        elif mode == 'LSB':
            lo, hi = center - 0.003, center - 0.0003
        else:
            h = half_bw.get(mode, 0.1)
            lo, hi = center - h, center + h
        for region in (self.bw_region, self.wf_bw_region):
            region.setRegion([lo, hi])
        for line in (self.vfo_line, self.wf_vfo_line):
            line.setPos(center)

    def _step_freq(self, direction):
        steps = {'1 kHz': 0.001, '5 kHz': 0.005, '10 kHz': 0.01,
                 '25 kHz': 0.025, '100 kHz': 0.1, '1 MHz': 1.0}
        step = steps.get(self.step_combo.currentText(), 0.01)
        self.freq_spin.setValue(self.freq_spin.value() + direction * step)

    def _toggle_fm(self, checked):
        if checked:
            self.deemph_state = 0.0
            self.audio_stream = sd.OutputStream(
                samplerate=AUDIO_RATE, channels=1, dtype='float32'
            )
            self.audio_stream.start()
        else:
            if self.audio_stream is not None:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None

    def _demodulate_fm(self, samples):
        demod = np.angle(samples[1:] * np.conj(samples[:-1])) / np.pi
        target_len = int(len(demod) * AUDIO_RATE / self.sample_rate)
        if target_len < 1:
            return
        audio = np.interp(
            np.linspace(0, len(demod) - 1, target_len),
            np.arange(len(demod)),
            demod,
        )
        if self.mode_combo.currentText() == 'NFM':
            # Low-pass to ~4 kHz for voice
            k = np.ones(12) / 12
            audio = np.convolve(audio, k, mode='same')
        else:
            # WFM: 75 µs de-emphasis
            alpha = np.exp(-1.0 / (75e-6 * AUDIO_RATE))
            out = np.empty_like(audio)
            state = self.deemph_state
            for i in range(len(audio)):
                state = (1.0 - alpha) * audio[i] + alpha * state
                out[i] = state
            self.deemph_state = state
            audio = out
        volume = self.vol_slider.value() / 100.0
        np.clip(audio * volume, -1.0, 1.0, out=audio)
        try:
            self.audio_stream.write(audio.astype(np.float32))
        except Exception:
            pass

    def _demodulate_am(self, samples):
        envelope = np.abs(samples)
        envelope -= envelope.mean()
        target_len = int(len(envelope) * AUDIO_RATE / self.sample_rate)
        if target_len < 1:
            return
        audio = np.interp(
            np.linspace(0, len(envelope) - 1, target_len),
            np.arange(len(envelope)),
            envelope,
        )
        volume = self.vol_slider.value() / 100.0
        np.clip(audio * volume * 4, -1.0, 1.0, out=audio)
        try:
            self.audio_stream.write(audio.astype(np.float32))
        except Exception:
            pass

    def update_waterfall_levels(self):
        min_lvl = self.min_lvl_slider.value()
        max_lvl = self.max_lvl_slider.value()
        
        # Prevent max from going lower than min
        if max_lvl <= min_lvl:
            max_lvl = min_lvl + 1
            self.max_lvl_slider.setValue(max_lvl)

        self.min_lvl_label.setText(f"Min Level: {min_lvl} dB")
        self.max_lvl_label.setText(f"Max Level: {max_lvl} dB")
        self.waterfall_image.setLevels([min_lvl, max_lvl])
        self.spectrum_plot.setYRange(min_lvl - 10, max_lvl + 10)

    def update_plots(self):
        if not self.playing:
            return

        read_size = AUDIO_CHUNK if self.audio_stream is not None else FFT_SIZE
        try:
            samples = self.sdr.read_samples(read_size)
        except Exception:
            return
        if len(samples) < FFT_SIZE:
            return

        power_spectrum = get_waterfall_bins(np.asarray(samples[:FFT_SIZE]))

        if self.audio_stream is not None:
            mode = self.mode_combo.currentText()
            raw = np.asarray(samples)
            if mode in ('WFM', 'NFM'):
                self._demodulate_fm(raw)
            elif mode == 'AM':
                self._demodulate_am(raw)

        # EMA averaging: alpha=1 is no averaging, lower alpha = more smoothing
        alpha = 1.0 / self.avg_slider.value()
        if self.avg_spectrum is None:
            self.avg_spectrum = power_spectrum.copy()
        else:
            self.avg_spectrum += alpha * (power_spectrum - self.avg_spectrum)

        # Update Spectrum
        self.spectrum_curve.setData(self.freqs_mhz, self.avg_spectrum)

        # Update Waterfall (SDRSharp style falls downwards)
        self.waterfall_data = np.roll(self.waterfall_data, -1, axis=1)
        self.waterfall_data[:, -1] = self.avg_spectrum
        self.waterfall_image.setImage(self.waterfall_data, autoLevels=False)

    def closeEvent(self, event):
        self.timer.stop()
        if self.audio_stream is not None:
            self.audio_stream.stop()
            self.audio_stream.close()
        self.sdr.close()
        event.accept()

if __name__ == '__main__':
    sdr = RtlSdr()
    app = QApplication(sys.argv)
    window = SDRWindow(sdr)
    window.show()
    sys.exit(app.exec())
