#!/usr/bin/env python3
"""
Zaawansowana aplikacja do analizy audio z dynamicznym EQ
EQ zmienia się automatycznie w czasie rzeczywistym zgodnie z krzywymi korekcyjnymi
+ tryb PipeWire/PulseAudio dla przetwarzania audio w czasie rzeczywistym
"""
import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Gst', '1.0')
gi.require_version('Gdk', '4.0')
from gi.repository import Gtk, Gst, GLib, GObject, Gdk
import cairo
import numpy as np
import threading
import subprocess
from scipy import signal
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import interp1d
from collections import deque
import math
import os
import time
from pydub import AudioSegment

# Inicjalizacja GStreamer
Gst.init(None)

def create_virtual_sink(sink_name="dynamic_eq_sink"):
    """Tworzy wirtualne wyjście audio za pomocą pactl."""
    try:
        subprocess.run(
            ["pactl", "load-module", "module-null-sink", f"sink_name={sink_name}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"Utworzono wirtualne wyjście audio: {sink_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Nie udało się utworzyć wirtualnego wyjścia: {e.stderr.decode('utf-8')}")
        return False

def remove_virtual_sink(sink_name="dynamic_eq_sink"):
    """Usuwa wirtualne wyjście audio."""
    try:
        subprocess.run(
            ["pactl", "unload-module", "module-null-sink"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"Usunięto wirtualne wyjście audio: {sink_name}")
    except subprocess.CalledProcessError as e:
        print(f"Błąd podczas usuwania wirtualnego wyjścia: {e}")

class DynamicEQController:
    """Kontroler dla dynamicznego EQ"""
    
    def __init__(self, equalizer, eq_freqs):
        self.equalizer = equalizer
        self.eq_freqs = eq_freqs
        self.correction_curves = {}
        self.time_stamps = []
        self.is_active = False
        self.start_time = 0
        self.interpolators = {}
        self.mode = "file"  # "file" lub "realtime"
        
        # Initialize band mapping (fix for missing attribute)
        self.band_to_eq_mapping = {
            'low': [0, 1, 2],          # 31, 62, 125 Hz
            'mid-low': [2, 3],         # 125, 250 Hz  
            'mid': [3, 4, 5],          # 250, 500, 1000 Hz
            'mid-high': [5, 6, 7],     # 1000, 2000, 4000 Hz
            'high': [7, 8, 9]          # 4000, 8000, 16000 Hz
        }
        
    def set_mode(self, mode):
        """Ustawia tryb pracy: 'file' lub 'realtime'"""
        self.mode = mode
        
    def load_correction_curves(self, time_stamps, correction_curves):
        """Ładuje krzywe korekcyjne i tworzy interpolatory"""
        self.time_stamps = np.array(time_stamps)
        self.correction_curves = correction_curves
        
        # Tworzenie interpolatorów dla każdego pasma
        for band_name, corrections in correction_curves.items():
            if len(corrections) > 1 and len(self.time_stamps) > 1:
                # Interpolator liniowy
                self.interpolators[band_name] = interp1d(
                    self.time_stamps, 
                    corrections, 
                    kind='linear',
                    bounds_error=False,
                    fill_value=(corrections[0], corrections[-1])
                )
    
    def apply_realtime_corrections(self, band_analysis):
        """Zastosuj korekcje w czasie rzeczywistym na podstawie analizy"""
        if not self.equalizer or self.mode != "realtime":
            return
            
        # Mapowanie pasm analizy na pasma EQ
        band_corrections = {
            'low': 0.0,
            'mid-low': 0.0, 
            'mid': 0.0,
            'mid-high': 0.0,
            'high': 0.0
        }
        
        # Oblicz korekcje na podstawie bieżącej analizy
        for band_name, metrics in band_analysis.items():
            if band_name not in band_corrections:
                continue
                
            correction = 0.0
            
            # Prosta heurystyka korekcji w czasie rzeczywistym
            mean_db = 20 * np.log10(metrics['mean'] + 1e-10)
            peak_db = 20 * np.log10(metrics['peak'] + 1e-10)
            
            # Rezonanse
            if peak_db - mean_db > 6.0:
                correction -= (peak_db - mean_db) * 0.3
                
            # Dziury
            if mean_db < -12.0:
                correction += abs(mean_db) * 0.2
                
            # Ostrość w wysokich pasmach
            if band_name in ['mid-high', 'high'] and metrics['std'] > metrics['mean'] * 1.5:
                correction -= 1.5
                
            # Zamulenie w niskich pasmach  
            if band_name in ['low', 'mid-low'] and metrics['energy'] > metrics['mean'] * 0.7:
                correction -= 1.0
                
            band_corrections[band_name] = np.clip(correction, -6, 6)
        
        # Zastosuj korekcje do EQ
        for band_name, correction in band_corrections.items():
            if band_name in self.band_to_eq_mapping:
                for eq_idx in self.band_to_eq_mapping[band_name]:
                    current_val = self.equalizer.get_property(f"band{eq_idx}")
                    # Łagodne przejście
                    new_val = current_val * 0.8 + correction * 0.2
                    new_val = np.clip(new_val, -12, 12)
                    self.equalizer.set_property(f"band{eq_idx}", float(new_val))
    
    def start_dynamic_eq(self):
        """Rozpoczyna dynamiczny EQ"""
        if self.mode == "file" and not self.interpolators:
            return False
            
        self.is_active = True
        self.start_time = time.time()
        
        if self.mode == "file":
            # Uruchom timer do aktualizacji EQ dla trybu pliku
            GLib.timeout_add(50, self.update_eq)
        
        return True
    
    def stop_dynamic_eq(self):
        """Zatrzymuje dynamiczny EQ"""
        self.is_active = False
        # Reset EQ do wartości neutralnych
        for i in range(10):
            if self.equalizer:
                self.equalizer.set_property(f"band{i}", 0.0)
    
    def update_eq(self):
        """Aktualizuje ustawienia EQ na podstawie aktualnego czasu (tryb pliku)"""
        if not self.is_active or self.mode != "file":
            return False
            
        current_time = time.time() - self.start_time
        
        # Interpoluj korekcje dla każdego pasma
        for band_name, interpolator in self.interpolators.items():
            try:
                correction = float(interpolator(current_time))
                
                # Zastosuj korekcję do odpowiednich pasm EQ
                if band_name in self.band_to_eq_mapping:
                    for eq_idx in self.band_to_eq_mapping[band_name]:
                        # Pobierz aktualną wartość i zastosuj wagę
                        weight = 1.0 / len(self.band_to_eq_mapping[band_name])
                        current_val = self.equalizer.get_property(f"band{eq_idx}")
                        new_val = current_val * (1 - weight) + correction * weight
                        new_val = np.clip(new_val, -12, 12)
                        
                        self.equalizer.set_property(f"band{eq_idx}", float(new_val))
                        
            except Exception as e:
                print(f"Błąd interpolacji dla pasma {band_name}: {e}")
        
        return self.is_active  # Kontynuuj timer jeśli aktywny

class AudioAnalyzer:
    """Klasa analizująca audio i wykrywająca niedoskonałości"""

    def __init__(self):
        self.sample_rate = 44100
        self.buffer_size = 4096
        self.freq_bands = {
            'low': (20, 250),          # Bass
            'mid-low': (250, 800),     # Low-mids
            'mid': (800, 3000),        # Mids
            'mid-high': (3000, 8000),  # High-mids
            'high': (8000, 20000)      # Highs
        }

        # Bufory dla analizy czasowej całego utworu
        self.time_analysis = {band: [] for band in self.freq_bands}
        self.correction_curves = {band: [] for band in self.freq_bands}
        self.time_stamps = []
        
        # Bufory dla trybu real-time
        self.realtime_buffers = {band: deque(maxlen=10) for band in self.freq_bands}
        
        # Parametry detekcji niedoskonałości
        self.detection_params = {
            'resonance_threshold': 6.0,     # dB powyżej średniej
            'null_threshold': -12.0,        # dB poniżej średniej
            'harshness_factor': 1.5,        # Współczynnik ostrości
            'muddiness_threshold': 0.7,     # Próg zamulenia
            'sensitivity': 1.0              # Czułość dynamicznego EQ
        }

        # Sztywne ustawienia korekcji
        self.fixed_corrections = {
            'low': 0.0,
            'mid-low': 0.0,
            'mid': 0.0,
            'mid-high': 0.0,
            'high': 0.0
        }

    def analyze_full_audio(self, audio_data, chunk_size=4096):
        """Analizuje cały utwór fragmentami z wygładzaniem"""
        self.time_analysis = {band: [] for band in self.freq_bands}
        self.correction_curves = {band: [] for band in self.freq_bands}
        self.time_stamps = []
        
        num_chunks = len(audio_data) // chunk_size
        
        # Bufory do wygładzania
        smoothing_buffer = {band: deque(maxlen=5) for band in self.freq_bands}
        
        for i in range(0, len(audio_data) - chunk_size, chunk_size):
            chunk = audio_data[i:i + chunk_size]
            timestamp = i / self.sample_rate
            self.time_stamps.append(timestamp)
            
            # Analiza spektralna fragmentu
            band_analysis, spectrum, freqs = self.analyze_spectrum(chunk)
            
            # Wykryj niedoskonałości dla fragmentu
            imperfections = self.detect_imperfections(band_analysis)
            
            # Zapisz analizę dla każdego pasma
            for band_name in self.freq_bands:
                if band_name in band_analysis:
                    self.time_analysis[band_name].append(band_analysis[band_name])
                    
                    # Oblicz korekcję dla tego fragmentu
                    correction = 0.0
                    if band_name in imperfections:
                        for issue in imperfections[band_name]:
                            correction += issue['correction'] * self.detection_params['sensitivity']
                    
                    # Dodaj sztywne korekcje
                    correction += self.fixed_corrections[band_name]
                    
                    # Wygładzanie korekcji
                    smoothing_buffer[band_name].append(correction)
                    smoothed_correction = np.mean(smoothing_buffer[band_name])
                    
                    self.correction_curves[band_name].append(smoothed_correction)
        
        return self.time_analysis, self.correction_curves

    def analyze_spectrum(self, audio_data):
        """Analizuje spektrum częstotliwości"""
        # Zastosuj okno Hamminga
        window = np.hamming(len(audio_data))
        windowed_data = audio_data * window

        # FFT
        spectrum = np.abs(rfft(windowed_data))
        freqs = rfftfreq(len(windowed_data), 1/self.sample_rate)

        # Analiza per band
        band_analysis = {}
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_spectrum = spectrum[mask]

            if len(band_spectrum) > 0:
                band_analysis[band_name] = {
                    'mean': np.mean(band_spectrum),
                    'peak': np.max(band_spectrum),
                    'std': np.std(band_spectrum),
                    'energy': np.sum(band_spectrum**2),
                    'centroid': np.sum(freqs[mask] * band_spectrum) / np.sum(band_spectrum)
                }
            else:
                band_analysis[band_name] = {
                    'mean': 0, 'peak': 0, 'std': 0, 'energy': 0, 'centroid': 0
                }

        return band_analysis, spectrum, freqs

    def detect_imperfections(self, band_analysis):
        """Wykrywa niedoskonałości w poszczególnych pasmach"""
        imperfections = {}

        for band_name, metrics in band_analysis.items():
            issues = []

            # Konwersja do dB
            mean_db = 20 * np.log10(metrics['mean'] + 1e-10)
            peak_db = 20 * np.log10(metrics['peak'] + 1e-10)

            # Detekcja rezonansów
            if peak_db - mean_db > self.detection_params['resonance_threshold']:
                issues.append({
                    'type': 'resonance',
                    'severity': (peak_db - mean_db) / self.detection_params['resonance_threshold'],
                    'frequency': metrics['centroid'],
                    'correction': -(peak_db - mean_db) * 0.5  # Łagodniejsza korekcja dla dynamicznego EQ
                })

            # Detekcja dziur częstotliwościowych
            if mean_db < self.detection_params['null_threshold']:
                issues.append({
                    'type': 'null',
                    'severity': abs(mean_db / self.detection_params['null_threshold']),
                    'frequency': metrics['centroid'],
                    'correction': abs(mean_db) * 0.3
                })

            # Detekcja ostrości (high-mids/highs)
            if band_name in ['mid-high', 'high']:
                if metrics['std'] / (metrics['mean'] + 1e-10) > self.detection_params['harshness_factor']:
                    issues.append({
                        'type': 'harshness',
                        'severity': metrics['std'] / (metrics['mean'] + 1e-10),
                        'frequency': metrics['centroid'],
                        'correction': -2.0
                    })

            # Detekcja zamulenia (low/mid-low)
            if band_name in ['low', 'mid-low']:
                if metrics['energy'] / (metrics['mean'] + 1e-10) > self.detection_params['muddiness_threshold']:
                    issues.append({
                        'type': 'muddiness',
                        'severity': metrics['energy'] / (metrics['mean'] + 1e-10),
                        'frequency': metrics['centroid'],
                        'correction': -1.5
                    })

            imperfections[band_name] = issues

        return imperfections

    def set_sensitivity(self, sensitivity):
        """Ustawia czułość dynamicznego EQ"""
        self.detection_params['sensitivity'] = sensitivity

    def set_fixed_correction(self, band, value):
        """Ustawia sztywną korekcję dla pasma"""
        if band in self.fixed_corrections:
            self.fixed_corrections[band] = value

class SpectrumWidget(Gtk.DrawingArea):
    """Widget do rysowania spektrum i krzywych EQ z pozycją czasową"""

    def __init__(self):
        super().__init__()
        self.set_size_request(800, 400)
        self.spectrum_data = None
        self.eq_curve = None
        self.imperfections = {}
        self.current_time = 0.0
        self.total_time = 0.0
        self.time_stamps = []
        self.correction_curves = {}
        self.set_draw_func(self.draw)

    def draw(self, area, cr, width, height):
        """Rysowanie przy użyciu Cairo"""
        # Tło
        cr.set_source_rgb(0.05, 0.05, 0.05)
        cr.rectangle(0, 0, width, height)
        cr.fill()

        # Siatka
        self.draw_grid(cr, width, height)

        # Krzywe korekcyjne dla każdego pasma
        if self.correction_curves and self.time_stamps:
            self.draw_correction_curves(cr, width, height)

        # Aktualna pozycja czasowa
        if self.total_time > 0:
            self.draw_time_position(cr, width, height)

    def draw_grid(self, cr, width, height):
        """Rysuje siatkę"""
        cr.set_source_rgba(0.2, 0.2, 0.2, 0.5)
        cr.set_line_width(1)

        # Linie poziome (dB)
        db_levels = [-12, -6, 0, 6, 12]
        for db in db_levels:
            y = height * (0.5 - db / 24)
            cr.move_to(0, y)
            cr.line_to(width, y)
            cr.stroke()

            # Etykiety dB
            cr.set_source_rgba(0.4, 0.4, 0.4, 1)
            cr.move_to(5, y - 3)
            cr.show_text(f"{db} dB")

        # Linie pionowe (czas)
        if self.total_time > 0:
            time_intervals = max(1, int(self.total_time / 10))
            for t in range(0, int(self.total_time) + 1, time_intervals):
                x = width * t / self.total_time
                cr.move_to(x, 0)
                cr.line_to(x, height)
                cr.stroke()

                # Etykiety czasu
                cr.set_source_rgba(0.4, 0.4, 0.4, 1)
                cr.move_to(x + 3, height - 5)
                cr.show_text(f"{t}s")

    def draw_correction_curves(self, cr, width, height):
        """Rysuje krzywe korekcyjne dla wszystkich pasm"""
        colors = {
            'low': (1.0, 0.3, 0.3, 0.8),       # Czerwony
            'mid-low': (1.0, 0.7, 0.2, 0.8),   # Pomarańczowy
            'mid': (0.3, 1.0, 0.3, 0.8),       # Zielony
            'mid-high': (0.3, 0.7, 1.0, 0.8),  # Niebieski
            'high': (0.8, 0.3, 1.0, 0.8)       # Fioletowy
        }

        for band_name, corrections in self.correction_curves.items():
            if not corrections or not self.time_stamps:
                continue

            color = colors.get(band_name, (0.5, 0.5, 0.5, 0.8))
            cr.set_source_rgba(*color)
            cr.set_line_width(2)

            # Rysuj krzywą
            for i, (time, correction) in enumerate(zip(self.time_stamps, corrections)):
                x = width * time / self.total_time if self.total_time > 0 else 0
                y = height * (0.5 - correction / 24)  # Skala ±12dB
                
                if i == 0:
                    cr.move_to(x, y)
                else:
                    cr.line_to(x, y)

            cr.stroke()

    def draw_time_position(self, cr, width, height):
        """Rysuje aktualną pozycję czasową"""
        if self.total_time <= 0:
            return

        x = width * self.current_time / self.total_time
        
        # Linia pozycji
        cr.set_source_rgba(1.0, 1.0, 1.0, 0.8)
        cr.set_line_width(2)
        cr.move_to(x, 0)
        cr.line_to(x, height)
        cr.stroke()

        # Znacznik góra
        cr.set_source_rgba(1.0, 1.0, 0.0, 1.0)
        cr.move_to(x, 0)
        cr.line_to(x - 5, 10)
        cr.line_to(x + 5, 10)
        cr.close_path()
        cr.fill()

    def update_correction_data(self, time_stamps, correction_curves):
        """Aktualizuje dane krzywych korekcyjnych"""
        self.time_stamps = time_stamps
        self.correction_curves = correction_curves
        if time_stamps:
            self.total_time = max(time_stamps)
        self.queue_draw()

    def update_current_time(self, current_time):
        """Aktualizuje aktualną pozycję czasową"""
        self.current_time = current_time
        self.queue_draw()

    def update_spectrum(self, spectrum, freqs):
        """Aktualizuje dane spektrum dla trybu real-time"""
        self.spectrum_data = spectrum
        self.queue_draw()

class DynamicEQControlWidget(Gtk.Box):
    """Widget kontroli dynamicznego EQ z trybami pracy"""

    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        
        # Wybór trybu pracy
        mode_frame = Gtk.Frame()
        mode_frame.set_label("Tryb pracy")
        mode_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        
        self.mode_file_radio = Gtk.CheckButton.new_with_label("Analiza pliku")
        self.mode_file_radio.set_active(True)
        mode_box.append(self.mode_file_radio)
        
        self.mode_realtime_radio = Gtk.CheckButton.new_with_label("Przetwarzanie w czasie rzeczywistym")
        self.mode_realtime_radio.set_group(self.mode_file_radio)
        self.mode_realtime_radio.connect("toggled", self.on_mode_changed)
        mode_box.append(self.mode_realtime_radio)
        
        mode_frame.set_child(mode_box)
        self.append(mode_frame)
        
        # Status dynamicznego EQ
        self.status_label = Gtk.Label(label="Dynamiczny EQ: Nieaktywny")
        self.append(self.status_label)
        
        # Przyciski kontroli
        controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        
        self.enable_button = Gtk.Button(label="Włącz Dynamiczny EQ")
        self.enable_button.connect("clicked", self.on_enable_clicked)
        self.enable_button.set_sensitive(False)
        controls.append(self.enable_button)
        
        self.disable_button = Gtk.Button(label="Wyłącz Dynamiczny EQ")
        self.disable_button.connect("clicked", self.on_disable_clicked)
        self.disable_button.set_sensitive(False)
        controls.append(self.disable_button)
        
        # Przycisk trybu real-time
        self.realtime_button = Gtk.Button(label="Rozpocznij nasłuchiwanie")
        self.realtime_button.connect("clicked", self.on_realtime_clicked)
        self.realtime_button.set_sensitive(False)
        controls.append(self.realtime_button)
        
        self.append(controls)
        
        # Suwak czułości
        sensitivity_frame = Gtk.Frame()
        sensitivity_frame.set_label("Czułość Dynamicznego EQ")
        
        sensitivity_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        
        self.sensitivity_slider = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
        self.sensitivity_slider.set_range(0.1, 3.0)
        self.sensitivity_slider.set_value(1.0)
        self.sensitivity_slider.set_draw_value(True)
        self.sensitivity_slider.connect("value-changed", self.on_sensitivity_changed)
        
        sensitivity_box.append(Gtk.Label(label="Czułość:"))
        sensitivity_box.append(self.sensitivity_slider)
        
        sensitivity_frame.set_child(sensitivity_box)
        self.append(sensitivity_frame)
        
        # Callback referencje
        self.enable_callback = None
        self.disable_callback = None
        self.sensitivity_callback = None
        self.realtime_callback = None
        self.mode_callback = None
        
        self.current_mode = "file"
        self.realtime_active = False

    def on_mode_changed(self, widget):
        """Zmiana trybu pracy"""
        if self.mode_realtime_radio.get_active():
            self.current_mode = "realtime"
            self.enable_button.set_sensitive(False)
            self.realtime_button.set_sensitive(True)
            self.realtime_button.set_label("Rozpocznij nasłuchiwanie")
        else:
            self.current_mode = "file"
            self.realtime_button.set_sensitive(False)
            if hasattr(self, '_file_ready') and self._file_ready:
                self.enable_button.set_sensitive(True)
                
        if self.mode_callback:
            self.mode_callback(self.current_mode)

    def set_callbacks(self, enable_cb, disable_cb, sensitivity_cb, realtime_cb=None, mode_cb=None):
        """Ustawia callbacki"""
        self.enable_callback = enable_cb
        self.disable_callback = disable_cb
        self.sensitivity_callback = sensitivity_cb
        self.realtime_callback = realtime_cb
        self.mode_callback = mode_cb

    def on_enable_clicked(self, widget):
        """Włączenie dynamicznego EQ"""
        if self.enable_callback:
            success = self.enable_callback()
            if success:
                self.status_label.set_text("Dynamiczny EQ: Aktywny")
                self.enable_button.set_sensitive(False)
                self.disable_button.set_sensitive(True)

    def on_disable_clicked(self, widget):
        """Wyłączenie dynamicznego EQ"""
        if self.disable_callback:
            self.disable_callback()
            self.status_label.set_text("Dynamiczny EQ: Nieaktywny")
            if self.current_mode == "file":
                self.enable_button.set_sensitive(True)
            self.disable_button.set_sensitive(False)

    def on_realtime_clicked(self, widget):
        """Obsługa trybu real-time"""
        if not self.realtime_active:
            if self.realtime_callback:
                success = self.realtime_callback(True)
                if success:
                    self.realtime_active = True
                    self.realtime_button.set_label("Zatrzymaj nasłuchiwanie")
                    self.status_label.set_text("Nasłuchiwanie aktywne - EQ w czasie rzeczywistym")
        else:
            if self.realtime_callback:
                self.realtime_callback(False)
                self.realtime_active = False
                self.realtime_button.set_label("Rozpocznij nasłuchiwanie")
                self.status_label.set_text("Nasłuchiwanie zatrzymane")

    def on_sensitivity_changed(self, slider):
        """Zmiana czułości"""
        if self.sensitivity_callback:
            value = slider.get_value()
            self.sensitivity_callback(value)

    def set_ready(self, ready, mode="file"):
        """Ustawia stan gotowości"""
        self._file_ready = ready
        if mode == "file" and self.current_mode == "file":
            self.enable_button.set_sensitive(ready and self.status_label.get_text() == "Dynamiczny EQ: Nieaktywny")
        elif mode == "realtime" and self.current_mode == "realtime":
            self.realtime_button.set_sensitive(ready)

    def get_current_mode(self):
        """Zwraca aktualny tryb pracy"""
        return self.current_mode

class AudioAnalyzerApp(Gtk.ApplicationWindow):
    """Główne okno aplikacji z dynamicznym EQ"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_title("Audio Analyzer z Dynamicznym EQ + PipeWire/PulseAudio")
        self.set_default_size(1200, 800)

        self.analyzer = AudioAnalyzer()
        self.pipeline = None
        self.realtime_pipeline = None
        self.current_file = None
        self.is_playing = False
        self.is_realtime_active = False
        self.dynamic_eq_controller = None
        self.virtual_sink_created = False
        
        # Initialize rt_equalizer to None to fix AttributeError
        self.rt_equalizer = None

        self.setup_ui()
        self.setup_gstreamer()

    def setup_ui(self):
        """Tworzy interfejs użytkownika"""
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        main_box.set_margin_top(10)
        main_box.set_margin_bottom(10)
        main_box.set_margin_start(10)
        main_box.set_margin_end(10)
        self.set_child(main_box)

        # Pasek narzędzi
        toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        main_box.append(toolbar)

        # Przyciski kontroli pliku
        self.file_button = Gtk.Button(label="Wybierz plik audio")
        self.file_button.connect("clicked", self.on_file_choose)
        toolbar.append(self.file_button)

        self.play_button = Gtk.Button(label="Odtwórz")
        self.play_button.connect("clicked", self.on_play_pause)
        self.play_button.set_sensitive(False)
        toolbar.append(self.play_button)

        self.analyze_button = Gtk.Button(label="Analizuj dla Dynamicznego EQ")
        self.analyze_button.connect("clicked", self.on_analyze)
        self.analyze_button.set_sensitive(False)
        toolbar.append(self.analyze_button)

        # Status
        self.status_label = Gtk.Label(label="Gotowy")
        toolbar.append(self.status_label)

        # Widget spektrum z krzywymi czasowymi
        self.spectrum_widget = SpectrumWidget()
        main_box.append(self.spectrum_widget)

        # Widget kontroli dynamicznego EQ
        self.dynamic_eq_widget = DynamicEQControlWidget()
        self.dynamic_eq_widget.set_callbacks(
            self.enable_dynamic_eq,
            self.disable_dynamic_eq,
            self.set_eq_sensitivity,
            self.toggle_realtime_mode,
            self.on_mode_changed
        )
        main_box.append(self.dynamic_eq_widget)

        # Panel EQ do monitorowania
        eq_frame = Gtk.Frame()
        eq_frame.set_label("Monitor EQ (tylko odczyt)")
        main_box.append(eq_frame)

        eq_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        eq_frame.set_child(eq_box)

        self.eq_monitors = []
        eq_freqs = [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]

        for freq in eq_freqs:
            vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)

            # Monitor (tylko wyświetlanie)
            monitor = Gtk.ProgressBar()
            monitor.set_orientation(Gtk.Orientation.VERTICAL)
            monitor.set_size_request(30, 100)
            monitor.set_fraction(0.5)  # Środek = 0dB
            monitor.set_inverted(True)

            label = Gtk.Label(label=f"{freq}Hz")
            label.set_size_request(30, -1)

            vbox.append(monitor)
            vbox.append(label)
            eq_box.append(vbox)

            self.eq_monitors.append(monitor)

        # Timer do aktualizacji monitorów EQ
        GLib.timeout_add(100, self.update_eq_monitors)

    def setup_gstreamer(self):
        """Konfiguruje pipeline GStreamer"""
        # Pipeline dla odtwarzania plików
        self.pipeline = Gst.Pipeline.new("audio-pipeline")

        # Elementy
        self.filesrc = Gst.ElementFactory.make("filesrc", "source")
        self.decode = Gst.ElementFactory.make("decodebin", "decode")
        self.convert = Gst.ElementFactory.make("audioconvert", "convert")
        self.equalizer = Gst.ElementFactory.make("equalizer-10bands", "eq")
        self.sink = Gst.ElementFactory.make("autoaudiosink", "sink")

        # Dodaj elementy do pipeline
        for element in [self.filesrc, self.decode, self.convert, self.equalizer, self.sink]:
            if element:
                self.pipeline.add(element)

        # Łączenie elementów
        self.filesrc.link(self.decode)
        self.convert.link(self.equalizer)
        self.equalizer.link(self.sink)

        # Callback dla decodebin
        self.decode.connect("pad-added", self.on_pad_added)

        # Bus dla wiadomości
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_bus_message)

        # Stwórz kontroler dynamicznego EQ
        eq_freqs = [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        self.dynamic_eq_controller = DynamicEQController(self.equalizer, eq_freqs)

    def setup_realtime_pipeline(self):
        """Konfiguruje pipeline dla trybu real-time"""
        if self.realtime_pipeline:
            return True
            
        self.realtime_pipeline = Gst.Pipeline.new("realtime-pipeline")

        # Elementy dla trybu real-time
        self.rt_src = Gst.ElementFactory.make("pulsesrc", "rt_source")
        self.rt_convert = Gst.ElementFactory.make("audioconvert", "rt_convert")
        self.rt_equalizer = Gst.ElementFactory.make("equalizer-10bands", "rt_eq")
        self.rt_spectrum = Gst.ElementFactory.make("spectrum", "rt_spectrum")
        self.rt_sink = Gst.ElementFactory.make("pulsesink", "rt_sink")

        if not all([self.rt_src, self.rt_convert, self.rt_equalizer, self.rt_spectrum, self.rt_sink]):
            self.status_label.set_text("Błąd: Nie można utworzyć elementów GStreamer")
            return False

        # Konfiguracja spectrum
        self.rt_spectrum.set_property("bands", 1024)
        self.rt_spectrum.set_property("threshold", -80)
        self.rt_spectrum.set_property("interval", 100000000)  # 100ms

        # Konfiguracja sink - wirtualne wyjście
        if self.virtual_sink_created:
            self.rt_sink.set_property("device", "dynamic_eq_sink")

        # Dodaj elementy do pipeline
        for element in [self.rt_src, self.rt_convert, self.rt_equalizer, self.rt_spectrum, self.rt_sink]:
            self.realtime_pipeline.add(element)

        # Łączenie elementów
        self.rt_src.link(self.rt_convert)
        self.rt_convert.link(self.rt_equalizer)
        self.rt_equalizer.link(self.rt_spectrum)
        self.rt_spectrum.link(self.rt_sink)

        # Bus dla wiadomości real-time
        rt_bus = self.realtime_pipeline.get_bus()
        rt_bus.add_signal_watch()
        rt_bus.connect("message", self.on_realtime_bus_message)

        # Aktualizuj kontroler EQ dla trybu real-time
        self.dynamic_eq_controller.equalizer = self.rt_equalizer
        self.dynamic_eq_controller.set_mode("realtime")

        return True

    def on_realtime_bus_message(self, bus, message):
        """Obsługa wiadomości z pipeline real-time"""
        if message.type == Gst.MessageType.ELEMENT:
            struct = message.get_structure()
            if struct and struct.get_name() == "spectrum":
                try:
                    magnitudes = struct.get_value("magnitude")
                    if magnitudes:
                        spectrum = np.array(magnitudes, dtype=np.float32)
                        freqs = np.linspace(0, self.analyzer.sample_rate / 2, len(spectrum))
                        
                        # Konwertuj spectrum na dane audio dla analizy
                        # To jest uproszczone - w rzeczywistości spectrum to już FFT
                        band_analysis, _, _ = self.analyzer.analyze_spectrum(spectrum[:1024])
                        
                        # Zastosuj korekcje w czasie rzeczywistym
                        if self.dynamic_eq_controller.is_active:
                            self.dynamic_eq_controller.apply_realtime_corrections(band_analysis)
                        
                        # Aktualizuj UI
                        GLib.idle_add(self.spectrum_widget.update_spectrum, spectrum, freqs)
                        
                except Exception as e:
                    print(f"Błąd przetwarzania spektrum: {e}")

        elif message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Błąd pipeline real-time: {err}, {debug}")
            GLib.idle_add(self.status_label.set_text, f"Błąd real-time: {err}")

    def on_mode_changed(self, mode):
        """Callback zmiany trybu pracy"""
        if mode == "realtime":
            if not self.virtual_sink_created:
                self.virtual_sink_created = create_virtual_sink()
                if not self.virtual_sink_created:
                    self.status_label.set_text("Błąd: Nie można utworzyć wirtualnego wyjścia")
                    return
                    
            # Zatrzymaj odtwarzanie pliku jeśli aktywne
            if self.is_playing:
                self.pipeline.set_state(Gst.State.PAUSED)
                self.play_button.set_label("Odtwórz")
                self.is_playing = False
                
            self.dynamic_eq_widget.set_ready(True, "realtime")
            self.status_label.set_text("Tryb real-time - gotowy do nasłuchiwania")
        else:
            # Tryb pliku
            if self.is_realtime_active:
                self.stop_realtime_mode()
            self.status_label.set_text("Tryb pliku - wybierz plik audio")

    def toggle_realtime_mode(self, start):
        """Włącza/wyłącza tryb real-time"""
        if start:
            return self.start_realtime_mode()
        else:
            self.stop_realtime_mode()
            return True

    def start_realtime_mode(self):
        """Rozpoczyna nasłuchiwanie w trybie real-time"""
        if not self.setup_realtime_pipeline():
            return False

        # Uruchom pipeline real-time
        ret = self.realtime_pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self.status_label.set_text("Błąd: Nie można uruchomić nasłuchiwania")
            return False

        self.is_realtime_active = True
        
        # Włącz automatycznie dynamiczny EQ w trybie real-time
        self.dynamic_eq_controller.set_mode("realtime")
        self.dynamic_eq_controller.start_dynamic_eq()
        
        return True

    def stop_realtime_mode(self):
        """Zatrzymuje nasłuchiwanie w trybie real-time"""
        if self.realtime_pipeline:
            self.realtime_pipeline.set_state(Gst.State.NULL)
        
        if self.dynamic_eq_controller:
            self.dynamic_eq_controller.stop_dynamic_eq()
            
        self.is_realtime_active = False

    def on_pad_added(self, element, pad):
        """Callback gdy decodebin utworzy pad"""
        sink_pad = self.convert.get_static_pad("sink")
        if not sink_pad.is_linked():
            pad.link(sink_pad)

    def on_bus_message(self, bus, message):
        """Obsługa wiadomości z GStreamer"""
        if message.type == Gst.MessageType.EOS:
            self.pipeline.set_state(Gst.State.READY)
            self.is_playing = False
            self.dynamic_eq_controller.stop_dynamic_eq()
            GLib.idle_add(self.play_button.set_label, "Odtwórz")

        elif message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, {debug}")
            self.pipeline.set_state(Gst.State.NULL)
            self.dynamic_eq_controller.stop_dynamic_eq()

    def on_file_choose(self, widget):
        """Wybór pliku audio"""
        dialog = Gtk.FileChooserDialog(
            title="Wybierz plik audio",
            transient_for=self,
            modal=True,
            action=Gtk.FileChooserAction.OPEN
        )
        dialog.add_buttons("Anuluj", Gtk.ResponseType.CANCEL, "Otwórz", Gtk.ResponseType.OK)

        filter_audio = Gtk.FileFilter()
        filter_audio.set_name("Pliki audio")
        filter_audio.add_mime_type("audio/*")
        dialog.add_filter(filter_audio)

        dialog.connect("response", self.on_file_dialog_response)
        dialog.present()

    def on_file_dialog_response(self, dialog, response):
        """Obsługa wyboru pliku"""
        if response == Gtk.ResponseType.OK:
            file = dialog.get_file()
            self.current_file = file.get_path()
            self.filesrc.set_property("location", self.current_file)

            self.status_label.set_text(f"Załadowano: {os.path.basename(self.current_file)}")
            self.play_button.set_sensitive(True)
            self.analyze_button.set_sensitive(True)

        dialog.destroy()

    def on_play_pause(self, widget):
        """Odtwarzanie/pauza z synchronizacją dynamicznego EQ"""
        if not self.is_playing:
            self.pipeline.set_state(Gst.State.PLAYING)
            self.play_button.set_label("Pauza")
            self.is_playing = True
            
            # Uruchom timer pozycji czasowej
            GLib.timeout_add(100, self.update_time_position)
            
        else:
            self.pipeline.set_state(Gst.State.PAUSED)
            self.play_button.set_label("Odtwórz")
            self.is_playing = False

    def update_time_position(self):
        """Aktualizuje pozycję czasową w spektrum"""
        if not self.is_playing:
            return False
            
        # Pobierz aktualną pozycję z pipeline
        success, position = self.pipeline.query_position(Gst.Format.TIME)
        if success:
            current_time = position / Gst.SECOND
            self.spectrum_widget.update_current_time(current_time)
        
        return self.is_playing

    def on_analyze(self, widget):
        """Przeprowadza analizę dla dynamicznego EQ"""
        if not self.current_file:
            return

        self.status_label.set_text("Analizowanie dla dynamicznego EQ...")

        # Analiza w osobnym wątku
        thread = threading.Thread(target=self.analyze_for_dynamic_eq)
        thread.start()

    def analyze_for_dynamic_eq(self):
        """Analizuje plik dla dynamicznego EQ"""
        try:
            # Wczytaj plik audio
            audio = AudioSegment.from_file(self.current_file)
            
            # Konwertuj na mono jeśli stereo
            if audio.channels == 2:
                audio = audio.set_channels(1)
            
            # Pobierz dane audio
            samples = np.array(audio.get_array_of_samples())
            audio_data = samples.astype(np.float32) / (2**15)

            # Analizuj z mniejszymi fragmentami dla lepszej responsywności
            time_analysis, correction_curves = self.analyzer.analyze_full_audio(audio_data, chunk_size=2048)

            # Aktualizuj UI
            GLib.idle_add(self.update_dynamic_eq_analysis, time_analysis, correction_curves)

        except Exception as e:
            GLib.idle_add(self.status_label.set_text, f"Błąd analizy: {str(e)}")

    def update_dynamic_eq_analysis(self, time_analysis, correction_curves):
        """Aktualizuje wyniki analizy dla dynamicznego EQ"""
        # Aktualizuj widget spektrum z krzywymi
        self.spectrum_widget.update_correction_data(self.analyzer.time_stamps, correction_curves)

        # Załaduj krzywe do kontrolera dynamicznego EQ
        self.dynamic_eq_controller.load_correction_curves(self.analyzer.time_stamps, correction_curves)
        self.dynamic_eq_controller.set_mode("file")

        self.status_label.set_text("Analiza zakończona - gotowy do dynamicznego EQ")
        self.dynamic_eq_widget.set_ready(True, "file")

        # Zapisz dane
        self.last_time_analysis = time_analysis
        self.last_correction_curves = correction_curves

    def enable_dynamic_eq(self):
        """Włącza dynamiczny EQ"""
        if not self.dynamic_eq_controller:
            return False
            
        success = self.dynamic_eq_controller.start_dynamic_eq()
        if success:
            self.status_label.set_text("Dynamiczny EQ aktywny")
        else:
            self.status_label.set_text("Błąd: Brak danych do dynamicznego EQ")
        return success

    def disable_dynamic_eq(self):
        """Wyłącza dynamiczny EQ"""
        if self.dynamic_eq_controller:
            self.dynamic_eq_controller.stop_dynamic_eq()
            self.status_label.set_text("Dynamiczny EQ wyłączony")

    def set_eq_sensitivity(self, sensitivity):
        """Ustawia czułość dynamicznego EQ"""
        self.analyzer.set_sensitivity(sensitivity)
        
        # Jeśli mamy dane, przeładuj z nową czułością
        if hasattr(self, 'last_time_analysis'):
            thread = threading.Thread(target=self.reanalyze_with_sensitivity)
            thread.start()

    def reanalyze_with_sensitivity(self):
        """Ponownie analizuje z nową czułością"""
        try:
            # Wczytaj plik ponownie
            audio = AudioSegment.from_file(self.current_file)
            if audio.channels == 2:
                audio = audio.set_channels(1)
            
            samples = np.array(audio.get_array_of_samples())
            audio_data = samples.astype(np.float32) / (2**15)

            # Analizuj z nową czułością
            time_analysis, correction_curves = self.analyzer.analyze_full_audio(audio_data, chunk_size=2048)

            # Aktualizuj kontroler dynamicznego EQ
            GLib.idle_add(lambda: self.dynamic_eq_controller.load_correction_curves(
                self.analyzer.time_stamps, correction_curves))
            
            # Aktualizuj wizualizację
            GLib.idle_add(lambda: self.spectrum_widget.update_correction_data(
                self.analyzer.time_stamps, correction_curves))

        except Exception as e:
            print(f"Błąd ponownej analizy: {e}")

    def update_eq_monitors(self):
        """Aktualizuje monitory EQ"""
        equalizer = None
        
        # Wybierz odpowiedni equalizer w zależności od trybu
        if self.dynamic_eq_widget.get_current_mode() == "realtime" and self.rt_equalizer:
            equalizer = self.rt_equalizer
        elif self.equalizer:
            equalizer = self.equalizer
            
        if not equalizer:
            return True
            
        for i, monitor in enumerate(self.eq_monitors):
            try:
                # Pobierz aktualną wartość pasma EQ
                gain = equalizer.get_property(f"band{i}")
                
                # Konwertuj na ułamek dla ProgressBar (0.0-1.0)
                # -12dB = 0.0, 0dB = 0.5, +12dB = 1.0
                fraction = (gain + 12) / 24
                fraction = max(0.0, min(1.0, fraction))
                
                monitor.set_fraction(fraction)
                
            except Exception as e:
                print(f"Błąd aktualizacji monitora {i}: {e}")
        
        return True

    def on_close(self):
        """Obsługa zamykania aplikacji"""
        # Zatrzymaj wszystkie pipeline
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        if self.realtime_pipeline:
            self.realtime_pipeline.set_state(Gst.State.NULL)
            
        # Usuń wirtualne wyjście
        if self.virtual_sink_created:
            remove_virtual_sink()

def main():
    """Funkcja główna"""
    # Ustaw style CSS dla kolorowania monitorów EQ
    css_provider = Gtk.CssProvider()
    css_provider.load_from_data(b"""
        .error { background: linear-gradient(to top, #ff4444, #ff6666); }
        .warning { background: linear-gradient(to top, #ffaa00, #ffcc44); }
        .accent { background: linear-gradient(to top, #4444ff, #6666ff); }
    """)
    
    # Poprawka dla GTK4 - użyj display z default screen
    display = Gdk.Display.get_default()
    Gtk.StyleContext.add_provider_for_display(
        display,
        css_provider,
        Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
    )
    
    app = Gtk.Application(application_id="org.example.dynamiceqanalyzer")
    
    def on_activate(application):
        window = AudioAnalyzerApp(application=application)
        # Połącz sygnał zamykania
        window.connect("close-request", lambda w: window.on_close())
        window.present()
    
    app.connect("activate", on_activate)
    app.run(None)

if __name__ == "__main__":
    main()