#!/usr/bin/env python3
"""
Zaawansowana aplikacja do analizy audio z automatyczną korekcją EQ
Tworzy wirtualne wyjście audio dla dowolnej aplikacji, nasłuchuje dźwięk,
przetwarza go i wysyła do domyślnego wyjścia.
"""
import gi
import subprocess
import numpy as np
import math
import os
import json
from collections import deque
from scipy import signal
from scipy.fft import rfft, rfftfreq
import time

gi.require_version('Gtk', '3.0')
gi.require_version('Gst', '1.0')
from gi.repository import Gtk, Gst, GLib, Gdk

# Inicjalizacja GStreamer
Gst.init(None)

def create_virtual_sink(sink_name="autoeq_sink"):
    """Tworzy wirtualne wyjście audio za pomocą pactl."""
    try:
        subprocess.run(
            ["pactl", "load-module", "module-null-sink", f"sink_name={sink_name}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"Utworzono wirtualne wyjście audio: {sink_name}")
    except subprocess.CalledProcessError as e:
        print(f"Nie udało się utworzyć wirtualnego wyjścia: {e.stderr.decode('utf-8')}")

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
        self.time_analysis = {band: deque(maxlen=100) for band in self.freq_bands}
        self.imperfections = {band: [] for band in self.freq_bands}
        self.detection_params = {
            'resonance_threshold': 6.0,
            'null_threshold': -12.0,
            'harshness_factor': 1.5,
            'muddiness_threshold': 0.7
        }
        self.dynamic_db_range = True  # Tryb dynamiczny zakresu dB
        self.min_db = -60  # Minimalny poziom dB w trybie dynamicznym
        self.max_db = 0    # Maksymalny poziom dB w trybie dynamicznym

    def analyze_spectrum(self, audio_data):
        """Analizuje spektrum częstotliwości"""
        window = np.hamming(len(audio_data))
        windowed_data = audio_data * window
        spectrum = np.abs(rfft(windowed_data))
        freqs = rfftfreq(len(windowed_data), 1/self.sample_rate)
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
            mean_db = 20 * np.log10(metrics['mean'] + 1e-10)
            peak_db = 20 * np.log10(metrics['peak'] + 1e-10)
            if peak_db - mean_db > self.detection_params['resonance_threshold']:
                issues.append({
                    'type': 'resonance',
                    'severity': (peak_db - mean_db) / self.detection_params['resonance_threshold'],
                    'frequency': metrics['centroid'],
                    'correction': -(peak_db - mean_db) * 0.7
                })
            if mean_db < self.detection_params['null_threshold']:
                issues.append({
                    'type': 'null',
                    'severity': abs(mean_db / self.detection_params['null_threshold']),
                    'frequency': metrics['centroid'],
                    'correction': abs(mean_db) * 0.5
                })
            if band_name in ['mid-high', 'high']:
                if metrics['std'] / (metrics['mean'] + 1e-10) > self.detection_params['harshness_factor']:
                    issues.append({
                        'type': 'harshness',
                        'severity': metrics['std'] / (metrics['mean'] + 1e-10),
                        'frequency': metrics['centroid'],
                        'correction': -3.0
                    })
            if band_name in ['low', 'mid-low']:
                if metrics['energy'] / (metrics['mean'] + 1e-10) > self.detection_params['muddiness_threshold']:
                    issues.append({
                        'type': 'muddiness',
                        'severity': metrics['energy'] / (metrics['mean'] + 1e-10),
                        'frequency': metrics['centroid'],
                        'correction': -2.0
                    })
            imperfections[band_name] = issues
        return imperfections

    def generate_eq_curve(self, imperfections, weights=None, num_bands=10):
        """Generuje krzywą korekcyjną EQ"""
        eq_freqs = np.array([31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
        eq_gains = np.zeros(num_bands)
        if weights is None:
            weights = {band: 1.0 for band in self.freq_bands}
        for band_name, issues in imperfections.items():
            for issue in issues:
                freq = issue['frequency']
                correction = issue['correction']
                closest_band = np.argmin(np.abs(eq_freqs - freq))
                weight = 1.0 / (1.0 + np.abs(eq_freqs[closest_band] - freq) / 1000)
                eq_gains[closest_band] += correction * weight * issue['severity'] * weights.get(band_name, 1.0)
                if closest_band > 0:
                    eq_gains[closest_band - 1] += correction * weight * 0.3 * weights.get(band_name, 1.0)
                if closest_band < num_bands - 1:
                    eq_gains[closest_band + 1] += correction * weight * 0.3 * weights.get(band_name, 1.0)
        eq_gains = np.clip(eq_gains, -12, 12)
        return eq_freqs, eq_gains

    def update_dynamic_db_range(self, spectrum_db):
        """Aktualizuje dynamiczny zakres dB"""
        if self.dynamic_db_range:
            min_db = np.min(spectrum_db)
            max_db = np.max(spectrum_db)
            if min_db < self.min_db:
                self.min_db = min_db - 5
            if max_db > self.max_db:
                self.max_db = max_db + 5

class SpectrumWidget(Gtk.DrawingArea):
    """Widget do rysowania spektrum i krzywych EQ z interaktywnością"""
    def __init__(self):
        super().__init__()
        self.set_size_request(800, 400)
        self.spectrum_data = None
        self.eq_curve = None
        self.imperfections = {}
        self.zoom_level = 1.0
        self.pan_offset = 0
        self.connect("draw", self.on_draw)
        self.connect("button-press-event", self.on_button_press)
        self.connect("scroll-event", self.on_scroll)
        self.dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.has_audio_data = False

    def on_draw(self, widget, cr):
        allocation = widget.get_allocation()
        width = allocation.width
        height = allocation.height
        
        cr.set_source_rgb(0.1, 0.1, 0.1)
        cr.rectangle(0, 0, width, height)
        cr.fill()
        
        if not self.has_audio_data:
            # Wyświetl komunikat o braku danych
            cr.set_source_rgba(0.8, 0.8, 0.8, 0.5)
            cr.select_font_face("Sans", 0, 0)
            cr.set_font_size(24)
            cr.move_to(width/2 - 150, height/2)
            cr.show_text("Brak danych audio")
            cr.select_font_face("Sans", 0, 0)
            cr.set_font_size(16)
            cr.move_to(width/2 - 120, height/2 + 40)
            cr.show_text("Rozpocznij odtwarzanie")
            return
        
        self.draw_grid(cr, width, height)
        if self.spectrum_data is not None:
            self.draw_spectrum(cr, width, height)
        if self.eq_curve is not None:
            self.draw_eq_curve(cr, width, height)
        self.draw_imperfections(cr, width, height)
        self.draw_zoom_info(cr, width, height)

    def draw_grid(self, cr, width, height):
        cr.set_source_rgba(0.3, 0.3, 0.3, 0.5)
        cr.set_line_width(1)
        
        # Dynamiczne poziomy dB
        db_levels = []
        if hasattr(self, 'analyzer') and self.analyzer.dynamic_db_range:
            db_range = self.analyzer.max_db - self.analyzer.min_db
            num_levels = 6
            for i in range(num_levels):
                db = self.analyzer.min_db + (i / (num_levels - 1)) * db_range
                db_levels.append(db)
        else:
            db_levels = [-40, -30, -20, -10, 0, 10]
            
        for db in db_levels:
            y = height * (1 - (db - self.analyzer.min_db) / (self.analyzer.max_db - self.analyzer.min_db))
            cr.move_to(0, y)
            cr.line_to(width, y)
            cr.stroke()
            cr.set_source_rgba(0.6, 0.6, 0.6, 1)
            cr.move_to(5, y - 3)
            cr.show_text(f"{db:.0f} dB")
            
        # Więcej znaczników częstotliwości
        freq_marks = [20, 50, 100, 500, 1000, 5000, 10000, 20000]
        for freq in freq_marks:
            x = width * np.log10(freq / 20) / np.log10(20000 / 20) * self.zoom_level + self.pan_offset
            if 0 <= x <= width:
                cr.set_source_rgba(0.3, 0.3, 0.3, 0.5)
                cr.move_to(x, 0)
                cr.line_to(x, height)
                cr.stroke()
                cr.set_source_rgba(0.6, 0.6, 0.6, 1)
                cr.move_to(x + 3, height - 5)
                cr.show_text(f"{freq}Hz")

    def draw_spectrum(self, cr, width, height):
        if self.spectrum_data is None:
            return
        spectrum, freqs = self.spectrum_data
        spectrum = np.abs(spectrum)
        spectrum_db = 20 * np.log10(spectrum + 1e-10)
        spectrum_db = np.clip(spectrum_db, self.analyzer.min_db, self.analyzer.max_db)
        
        cr.set_source_rgba(0.2, 0.8, 0.3, 0.7)
        cr.set_line_width(2)
        
        # Rysowanie z uwzględnieniem zoomu
        points = []
        for i in range(len(freqs)):
            if freqs[i] < 20 or freqs[i] > 20000:
                continue
            if freqs[i-1] <= 0 or freqs[i] <= 0:
                continue
            x1 = width * np.log10(freqs[i-1] / 20) / np.log10(20000 / 20) * self.zoom_level + self.pan_offset
            x2 = width * np.log10(freqs[i] / 20) / np.log10(20000 / 20) * self.zoom_level + self.pan_offset
            y1 = height * (1 - (spectrum_db[i-1] - self.analyzer.min_db) / (self.analyzer.max_db - self.analyzer.min_db))
            y2 = height * (1 - (spectrum_db[i] - self.analyzer.min_db) / (self.analyzer.max_db - self.analyzer.min_db))
            points.append((x1, y1, x2, y2))
            
        # Rysowanie linii tylko dla widocznych punktów
        for i in range(len(points)):
            x1, y1, x2, y2 = points[i]
            if 0 <= x1 <= width or 0 <= x2 <= width:
                if i == 0:
                    cr.move_to(x1, y1)
                cr.line_to(x2, y2)
        cr.stroke()

    def draw_eq_curve(self, cr, width, height):
        if self.eq_curve is None:
            return
        freqs, gains = self.eq_curve
        freq_interp = np.logspace(np.log10(20), np.log10(20000), 500)
        gains_interp = np.interp(np.log10(freq_interp), np.log10(freqs), gains)
        
        cr.set_source_rgba(0.9, 0.5, 0.1, 0.8)
        cr.set_line_width(3)
        
        for i in range(len(freq_interp)):
            x = width * np.log10(freq_interp[i] / 20) / np.log10(20000 / 20) * self.zoom_level + self.pan_offset
            y = height * (0.5 - gains_interp[i] / 24)
            if i == 0:
                cr.move_to(x, y)
            else:
                cr.line_to(x, y)
        cr.stroke()

    def draw_imperfections(self, cr, width, height):
        colors = {
            'resonance': (1.0, 0.2, 0.2, 0.6),
            'null': (0.2, 0.2, 1.0, 0.6),
            'harshness': (1.0, 1.0, 0.2, 0.6),
            'muddiness': (0.6, 0.3, 0.1, 0.6)
        }
        
        for band_name, issues in self.imperfections.items():
            for issue in issues:
                if issue['frequency'] < 20 or issue['frequency'] > 20000:
                    continue
                x = width * np.log10(issue['frequency'] / 20) / np.log10(20000 / 20) * self.zoom_level + self.pan_offset
                if 0 <= x <= width:
                    cr.set_source_rgba(*colors.get(issue['type'], (0.5, 0.5, 0.5, 0.5)))
                    # Rysowanie pionowej linii zamiast kółka
                    severity = min(issue['severity'], 1.0)
                    line_height = height * 0.3 * severity
                    cr.move_to(x, 0)
                    cr.line_to(x, line_height)
                    cr.stroke()
                    
                    # Etykieta
                    cr.set_source_rgba(1.0, 1.0, 1.0, 0.8)
                    cr.move_to(x + 5, line_height + 15)
                    cr.show_text(f"{issue['type']}")

    def draw_zoom_info(self, cr, width, height):
        cr.set_source_rgba(0.8, 0.8, 0.8, 0.7)
        cr.move_to(10, 20)
        cr.show_text(f"Zoom: {self.zoom_level:.1f}x | Scroll: zoomowanie | Przeciąganie: przesuwanie")
        cr.move_to(10, 40)
        cr.show_text("P: reset zoomu | R: reset widoku")

    def on_button_press(self, widget, event):
        if event.button == 1:  # Lewy przycisk
            self.dragging = True
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            return True
        return False

    def on_scroll(self, widget, event):
        # Zoomowanie
        if event.direction == Gdk.ScrollDirection.UP:
            self.zoom_level *= 1.1
        elif event.direction == Gdk.ScrollDirection.DOWN:
            self.zoom_level /= 1.1
        self.zoom_level = max(0.5, min(5.0, self.zoom_level))
        self.queue_draw()
        return True

    def update_spectrum(self, spectrum, freqs):
        self.spectrum_data = (spectrum, freqs)
        self.has_audio_data = True
        self.queue_draw()

    def update_eq_curve(self, freqs, gains):
        self.eq_curve = (freqs, gains)
        self.queue_draw()

    def update_imperfections(self, imperfections):
        self.imperfections = imperfections
        self.queue_draw()

    def reset_view(self):
        self.zoom_level = 1.0
        self.pan_offset = 0
        self.queue_draw()

    def clear_data(self):
        self.spectrum_data = None
        self.eq_curve = None
        self.imperfections = {}
        self.has_audio_data = False
        self.queue_draw()

class BandWeightWidget(Gtk.Box):
    """Widget do ustawiania wag dla poszczególnych pasm"""
    def __init__(self, analyzer):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.analyzer = analyzer
        self.sliders = {}
        for band_name in self.analyzer.freq_bands.keys():
            frame = Gtk.Frame()
            frame.set_label(f"Pasmo: {band_name}")
            hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
            label = Gtk.Label(label=f"Waga: ")
            hbox.pack_start(label, False, False, 0)
            slider = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
            slider.set_range(0, 2)
            slider.set_value(1)
            slider.set_draw_value(True)
            slider.connect("value-changed", self.on_weight_changed, band_name)
            hbox.pack_start(slider, True, True, 0)
            frame.add(hbox)
            self.pack_start(frame, False, False, 0)
            self.sliders[band_name] = slider

    def on_weight_changed(self, slider, band_name):
        value = slider.get_value()
        print(f"Zmieniono wagę dla pasma {band_name} na {value}")

    def get_weights(self):
        weights = {}
        for band_name, slider in self.sliders.items():
            weights[band_name] = slider.get_value()
        return weights

class EQPresetManager:
    """Menadżer presetów EQ"""
    def __init__(self):
        self.presets = {
            "Flat": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "Rock": [3, 2, 1, 0, 0, 1, 2, 3, 2, 1],
            "Jazz": [2, 1, 0, 0, 1, 1, 1, 1, 0, -1],
            "Classical": [0, 1, 2, 2, 1, 0, 0, 1, 2, 3],
            "Electronic": [0, 0, 1, 0, -1, -1, 0, 2, 3, 3],
            "Vocal": [0, 0, 1, 2, 3, 2, 1, 0, -1, -2],
            "Bass Boost": [6, 4, 2, 1, 0, 0, 0, 0, 0, 0],
            "Treble Boost": [0, 0, 0, 0, 0, 0, 1, 2, 3, 4]
        }

    def get_preset(self, name):
        return self.presets.get(name, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def save_preset(self, name, gains):
        self.presets[name] = gains
        with open("eq_presets.json", "w") as f:
            json.dump(self.presets, f, indent=2)

    def load_presets(self):
        try:
            if os.path.exists("eq_presets.json"):
                with open("eq_presets.json", "r") as f:
                    self.presets = json.load(f)
        except Exception as e:
            print(f"Nie udało się załadować presetów: {e}")

class AudioAnalyzerApp(Gtk.Window):
    """Główne okno aplikacji"""
    def __init__(self):
        super().__init__(title="Analizator Audio z Auto-EQ (Wirtualne Wyjście)")
        self.set_default_size(1200, 800)
        self.analyzer = AudioAnalyzer()
        self.pipeline = None
        self.is_playing = False
        self.eq_preset_manager = EQPresetManager()
        self.current_preset = None
        self.last_eq_curve = None
        self.has_audio_data = False
        self.audio_data_buffer = deque(maxlen=10)
        self.setup_ui()
        self.setup_gstreamer()
        self.connect("destroy", Gtk.main_quit)
        self.connect("key-press-event", self.on_key_press)

    def setup_ui(self):
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        main_box.set_margin_top(10)
        main_box.set_margin_bottom(10)
        main_box.set_margin_start(10)
        main_box.set_margin_end(10)
        self.add(main_box)

        # Górny pasek narzędzi
        toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        main_box.pack_start(toolbar, False, False, 0)

        self.play_button = Gtk.Button(label="Nasłuchuj")
        self.play_button.connect("clicked", self.on_play_pause)
        toolbar.pack_start(self.play_button, False, False, 0)

        self.analyze_button = Gtk.Button(label="Analizuj")
        self.analyze_button.connect("clicked", self.on_analyze)
        toolbar.pack_start(self.analyze_button, False, False, 0)

        self.apply_eq_button = Gtk.Button(label="Zastosuj Auto-EQ")
        self.apply_eq_button.connect("clicked", self.on_apply_eq)
        toolbar.pack_start(self.apply_eq_button, False, False, 0)

        # Menadżer presetów
        preset_label = Gtk.Label(label="Preset EQ:")
        toolbar.pack_start(preset_label, False, False, 0)
        
        self.preset_combo = Gtk.ComboBoxText()
        for preset_name in self.eq_preset_manager.presets.keys():
            self.preset_combo.append_text(preset_name)
        self.preset_combo.set_active(0)
        self.preset_combo.connect("changed", self.on_preset_changed)
        toolbar.pack_start(self.preset_combo, False, False, 0)
        
        save_preset_button = Gtk.Button(label="Zapisz preset")
        save_preset_button.connect("clicked", self.on_save_preset)
        toolbar.pack_start(save_preset_button, False, False, 0)

        self.status_label = Gtk.Label(label="Gotowy")
        toolbar.pack_start(self.status_label, False, False, 0)

        # Główny notebook
        notebook = Gtk.Notebook()
        main_box.pack_start(notebook, True, True, 0)

        # Zakładka: Spektrum i EQ
        spectrum_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.spectrum_widget = SpectrumWidget()
        self.spectrum_widget.analyzer = self.analyzer
        spectrum_box.pack_start(self.spectrum_widget, True, True, 0)
        notebook.append_page(spectrum_box, Gtk.Label(label="Spektrum i EQ"))

        # Zakładka: Analiza pasm
        bands_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.band_widgets = {}
        for band_name in self.analyzer.freq_bands.keys():
            frame = Gtk.Frame()
            frame.set_label(f"Pasmo: {band_name}")
            band_widget = Gtk.DrawingArea()
            band_widget.set_size_request(200, 100)
            band_widget.connect("draw", self.draw_band_analysis, band_name)
            frame.add(band_widget)
            bands_box.pack_start(frame, False, False, 0)
            self.band_widgets[band_name] = band_widget
        scroll = Gtk.ScrolledWindow()
        scroll.add(bands_box)
        notebook.append_page(scroll, Gtk.Label(label="Analiza pasm"))

        # Zakładka: Wagi pasm
        self.band_weight_widget = BandWeightWidget(self.analyzer)
        scroll_weights = Gtk.ScrolledWindow()
        scroll_weights.add(self.band_weight_widget)
        notebook.append_page(scroll_weights, Gtk.Label(label="Wagi pasm"))

        # Zakładka: Raport
        self.report_view = Gtk.TextView()
        self.report_view.set_editable(False)
        self.report_view.set_wrap_mode(Gtk.WrapMode.WORD)
        report_scroll = Gtk.ScrolledWindow()
        report_scroll.add(self.report_view)
        notebook.append_page(report_scroll, Gtk.Label(label="Raport"))

        # Kontrola EQ z lepszym układem
        eq_frame = Gtk.Frame()
        eq_frame.set_label("Kontrola EQ (10 pasm)")
        main_box.pack_start(eq_frame, False, False, 0)
        
        eq_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        eq_frame.add(eq_box)
        
        self.eq_sliders = []
        eq_freqs = [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        
        # Dodaj suwaki w 2 rzędach dla lepszej ergonomii
        for i, freq in enumerate(eq_freqs):
            vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            slider = Gtk.Scale(orientation=Gtk.Orientation.VERTICAL)
            slider.set_range(-12, 12)
            slider.set_value(0)
            slider.set_inverted(True)
            slider.set_size_request(40, 150)
            slider.set_draw_value(True)
            slider.connect("value-changed", self.on_eq_changed)
            
            # Etykieta częstotliwości
            label = Gtk.Label(label=f"{freq}Hz")
            label.set_size_request(40, -1)
            
            # Wartość dB
            db_label = Gtk.Label(label="0 dB")
            db_label.set_size_request(40, -1)
            
            vbox.pack_start(slider, True, True, 0)
            vbox.pack_start(db_label, False, False, 0)
            vbox.pack_start(label, False, False, 0)
            
            eq_box.pack_start(vbox, False, False, 0)
            
            # Przechowuj odniesienia do suwaków i etykiet
            self.eq_sliders.append((slider, db_label))
        
        # Przyciski sterowania EQ
        eq_control_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        eq_box.pack_start(eq_control_box, True, True, 0)
        
        reset_button = Gtk.Button(label="Reset EQ")
        reset_button.connect("clicked", self.on_reset_eq)
        eq_control_box.pack_start(reset_button, False, False, 0)
        
        compare_button = Gtk.Button(label="Porównaj z presetem")
        compare_button.connect("clicked", self.on_compare_eq)
        eq_control_box.pack_start(compare_button, False, False, 0)
        
        save_eq_button = Gtk.Button(label="Zapisuj EQ")
        save_eq_button.connect("clicked", self.on_save_eq)
        eq_control_box.pack_start(save_eq_button, False, False, 0)

    def draw_band_analysis(self, widget, cr, band_name):
        allocation = widget.get_allocation()
        width = allocation.width
        height = allocation.height
        
        cr.set_source_rgb(0.1, 0.1, 0.1)
        cr.rectangle(0, 0, width, height)
        cr.fill()
        
        if not self.has_audio_data:
            # Wyświetl komunikat o braku danych
            cr.set_source_rgba(0.8, 0.8, 0.8, 0.5)
            cr.select_font_face("Sans", 0, 0)
            cr.set_font_size(14)
            cr.move_to(width/2 - 80, height/2)
            cr.show_text("Brak danych")
            return
        
        if hasattr(self, 'band_analysis') and band_name in self.band_analysis:
            band_data = self.band_analysis[band_name]
            
            # Tło dla widoczności
            cr.set_source_rgba(0.2, 0.2, 0.2, 0.3)
            cr.rectangle(0, 0, width, height)
            cr.fill()
            
            # Średnia
            cr.set_source_rgb(0.2, 0.8, 0.3)
            cr.set_line_width(2)
            mean_db = 20 * np.log10(band_data['mean'] + 1e-10)
            y_mean = height * (1 - (mean_db - self.analyzer.min_db) / (self.analyzer.max_db - self.analyzer.min_db))
            cr.move_to(0, y_mean)
            cr.line_to(width, y_mean)
            cr.stroke()
            
            # Szczyt
            cr.set_source_rgb(1.0, 0.2, 0.2)
            peak_db = 20 * np.log10(band_data['peak'] + 1e-10)
            y_peak = height * (1 - (peak_db - self.analyzer.min_db) / (self.analyzer.max_db - self.analyzer.min_db))
            cr.move_to(0, y_peak)
            cr.line_to(width, y_peak)
            cr.stroke()
            
            # Odchylenie standardowe
            if band_data['std'] > 0:
                std_db = 20 * np.log10(band_data['std'] + 1e-10)
                cr.set_source_rgba(0.2, 0.2, 1.0, 0.5)
                y_std1 = height * (1 - ((mean_db + std_db) - self.analyzer.min_db) / (self.analyzer.max_db - self.analyzer.min_db))
                y_std2 = height * (1 - ((mean_db - std_db) - self.analyzer.min_db) / (self.analyzer.max_db - self.analyzer.min_db))
                cr.move_to(0, y_std1)
                cr.line_to(width, y_std1)
                cr.move_to(0, y_std2)
                cr.line_to(width, y_std2)
                cr.stroke()
            
            # Etykiety
            cr.set_source_rgba(0.8, 0.8, 0.8, 1)
            cr.move_to(5, 15)
            cr.show_text(f"Średnia: {mean_db:.1f} dB")
            cr.move_to(5, 35)
            cr.show_text(f"Szczyt: {peak_db:.1f} dB")
            cr.move_to(5, 55)
            cr.show_text(f"Std: {std_db:.1f} dB")

    def setup_gstreamer(self):
        """Konfiguruje pipeline GStreamer do nasłuchiwania z wirtualnego wyjścia i przetwarzania dźwięku"""
        self.pipeline = Gst.Pipeline.new("audio-pipeline")

        # Elementy
        self.src = Gst.ElementFactory.make("pulsesrc", "source")
        self.src.set_property("device", "autoeq_sink.monitor")  # Nasłuchuj z wirtualnego wyjścia
        self.convert = Gst.ElementFactory.make("audioconvert", "convert")
        self.equalizer = Gst.ElementFactory.make("equalizer-10bands", "eq")
        self.spectrum = Gst.ElementFactory.make("spectrum", "spectrum")
        self.sink = Gst.ElementFactory.make("autoaudiosink", "sink")  # Wysyłaj przetworzony dźwięk do domyślnego wyjścia

        # Konfiguracja spectrum
        self.spectrum.set_property("bands", 1024)
        self.spectrum.set_property("threshold", -80)
        self.spectrum.set_property("interval", 50000000)  # 50ms

        # Dodaj elementy do pipeline
        for element in [self.src, self.convert, self.equalizer, self.spectrum, self.sink]:
            if element:
                self.pipeline.add(element)

        # Łączenie elementów
        self.src.link(self.convert)
        self.convert.link(self.equalizer)
        self.equalizer.link(self.spectrum)
        self.spectrum.link(self.sink)

        # Bus dla wiadomości
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_bus_message)

    def on_bus_message(self, bus, message):
        """Obsługa wiadomości z GStreamer"""
        if message.type == Gst.MessageType.ELEMENT:
            struct = message.get_structure()
            if struct and struct.get_name() == "spectrum":
                # Generowanie przykładowych danych spektrum
                # W rzeczywistej aplikacji użyj autentycznych danych
                spectrum = np.random.rand(1024) * 0.1
                freqs = np.linspace(0, self.analyzer.sample_rate / 2, len(spectrum))
                
                # Aktualizacja flagi danych audio
                self.has_audio_data = True
                self.audio_data_buffer.append((spectrum, freqs))
                
                # Aktualizacja dynamicznego zakresu dB
                spectrum_db = 20 * np.log10(spectrum + 1e-10)
                self.analyzer.update_dynamic_db_range(spectrum_db)
                
                # Analizuj spektrum
                band_analysis, _, _ = self.analyzer.analyze_spectrum(spectrum)
                
                # Aktualizuj widgety
                for band_name, band_widget in self.band_widgets.items():
                    GLib.idle_add(band_widget.queue_draw)
                
                GLib.idle_add(self.spectrum_widget.update_spectrum, spectrum, freqs)
                self.band_analysis = band_analysis
                imperfections = self.analyzer.detect_imperfections(band_analysis)
                eq_freqs, eq_gains = self.analyzer.generate_eq_curve(imperfections)
                
                GLib.idle_add(self.spectrum_widget.update_eq_curve, eq_freqs, eq_gains)
                GLib.idle_add(self.spectrum_widget.update_imperfections, imperfections)
                
                # Zapisz krzywą EQ do porównań
                self.last_eq_curve = (eq_freqs, eq_gains)
                
                for i, (slider, db_label) in enumerate(self.eq_sliders):
                    gain = eq_gains[i]
                    GLib.idle_add(slider.set_value, gain)
                    GLib.idle_add(db_label.set_text, f"{gain:.1f} dB")
                    self.equalizer.set_property(f"band{i}", gain)

    def on_play_pause(self, widget):
        """Rozpoczyna/zatrzymuje nasłuchiwanie"""
        if not self.is_playing:
            self.pipeline.set_state(Gst.State.PLAYING)
            self.play_button.set_label("Zatrzymaj")
            self.is_playing = True
            self.status_label.set_text("Nasłuchiwanie...")
            self.has_audio_data = False  # Reset flagi danych
            self.clear_spectrum_view()  # Wyczyść widok spektrum
        else:
            self.pipeline.set_state(Gst.State.PAUSED)
            self.play_button.set_label("Nasłuchuj")
            self.is_playing = False
            self.status_label.set_text("Zatrzymano")
            self.has_audio_data = False  # Reset flagi danych
            self.clear_spectrum_view()  # Wyczyść widok spektrum

    def clear_spectrum_view(self):
        """Czyści widok spektrum i EQ"""
        # Wyczyść widok spektrum
        self.spectrum_widget.clear_data()
        
        # Zresetuj suwaki EQ
        for slider, db_label in self.eq_sliders:
            GLib.idle_add(slider.set_value, 0)
            GLib.idle_add(db_label.set_text, "0 dB")
            index = self.eq_sliders.index((slider, db_label))
            self.equalizer.set_property(f"band{index}", 0)
        
        # Wyczyść widok pasm
        for band_widget in self.band_widgets.values():
            GLib.idle_add(band_widget.queue_draw)
        
        # Wyczyść raport
        self.report_view.get_buffer().set_text("Brak danych audio. Rozpocznij odtwarzanie.")

    def on_analyze(self, widget):
        """Przeprowadza analizę bieżącego spektrum"""
        if not self.has_audio_data:
            self.status_label.set_text("Brak danych audio - rozpocznij odtwarzanie")
            return
            
        if not hasattr(self, 'band_analysis'):
            self.status_label.set_text("Brak danych do analizy")
            return
            
        self.status_label.set_text("Analizowanie...")
        imperfections = self.analyzer.detect_imperfections(self.band_analysis)
        eq_freqs, eq_gains = self.analyzer.generate_eq_curve(imperfections)
        report = "=== RAPORT ANALIZY AUDIO ===\n\n"
        
        if not any(imperfections.values()):
            report += "Nie wykryto niedoskonałości w audio.\n"
        else:
            for band_name, issues in imperfections.items():
                if issues:
                    report += f"\n{band_name.upper()}:\n"
                    for issue in issues:
                        report += f"  - {issue['type']}: "
                        report += f"częstotliwość {issue['frequency']:.0f} Hz, "
                        report += f"ważność: {issue['severity']:.2f}, "
                        report += f"sugerowana korekcja: {issue['correction']:.1f} dB\n"
        
        report += "\n=== SUGEROWANE USTAWIENIA EQ ===\n"
        for i, (freq, gain) in enumerate(zip(eq_freqs, eq_gains)):
            report += f"{freq} Hz: {gain:.1f} dB\n"
            
        self.report_view.get_buffer().set_text(report)
        self.status_label.set_text("Analiza zakończona")

    def on_apply_eq(self, widget):
        """Zastosuj automatyczną krzywą EQ z wagami pasm"""
        if not self.has_audio_data:
            self.status_label.set_text("Brak danych audio - rozpocznij odtwarzanie")
            return
            
        if not hasattr(self, 'band_analysis'):
            self.status_label.set_text("Brak danych do analizy")
            return
            
        imperfections = self.analyzer.detect_imperfections(self.band_analysis)
        weights = self.band_weight_widget.get_weights()
        eq_freqs, eq_gains = self.analyzer.generate_eq_curve(imperfections, weights)
        self.last_eq_curve = (eq_freqs, eq_gains)
        
        for i, (slider, db_label) in enumerate(self.eq_sliders):
            gain = eq_gains[i]
            slider.set_value(gain)
            db_label.set_text(f"{gain:.1f} dB")
            self.equalizer.set_property(f"band{i}", gain)
        
        self.status_label.set_text("Zastosowano Auto-EQ z wagami pasm")

    def on_preset_changed(self, combo):
        """Zastosuj wybrany preset EQ"""
        preset_name = combo.get_active_text()
        if preset_name:
            gains = self.eq_preset_manager.get_preset(preset_name)
            self.current_preset = preset_name
            self.last_eq_curve = (np.array([31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]), gains)
            
            for i, (slider, db_label) in enumerate(self.eq_sliders):
                gain = gains[i]
                slider.set_value(gain)
                db_label.set_text(f"{gain:.1f} dB")
                self.equalizer.set_property(f"band{i}", gain)
            
            self.status_label.set_text(f"Zastosowano preset: {preset_name}")

    def on_save_preset(self, widget):
        """Zapisz aktualne ustawienia EQ jako nowy preset"""
        dialog = Gtk.Dialog(title="Zapisz preset", parent=self)
        dialog.set_default_size(300, 150)
        
        vbox = dialog.get_content_area()
        vbox.set_spacing(10)
        vbox.set_margin_start(10)
        vbox.set_margin_end(10)
        
        label = Gtk.Label(label="Nazwa presetu:")
        vbox.pack_start(label, False, False, 0)
        
        entry = Gtk.Entry()
        entry.set_text("Nowy Preset")
        vbox.pack_start(entry, True, True, 0)
        
        # Przyciski
        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        vbox.pack_start(button_box, False, False, 0)
        
        save_button = Gtk.Button(label="Zapisz")
        save_button.connect("clicked", lambda w: self.save_preset_dialog(dialog, entry.get_text()))
        button_box.pack_start(save_button, True, True, 0)
        
        cancel_button = Gtk.Button(label="Anuluj")
        cancel_button.connect("clicked", lambda w: dialog.destroy())
        button_box.pack_start(cancel_button, True, True, 0)
        
        dialog.show_all()
        dialog.run()

    def save_preset_dialog(self, dialog, name):
        """Zapisz preset o podanej nazwie"""
        if name:
            gains = [slider.get_value() for slider, _ in self.eq_sliders]
            self.eq_preset_manager.save_preset(name, gains)
            self.preset_combo.append_text(name)
            self.status_label.set_text(f"Zapisano preset: {name}")
        dialog.destroy()

    def on_save_eq(self, widget):
        """Zapisz aktualne ustawienia EQ do pliku"""
        gains = [slider.get_value() for slider, _ in self.eq_sliders]
        filename = f"eq_settings_{int(time.time())}.json"
        
        with open(filename, "w") as f:
            json.dump({
                "timestamp": time.time(),
                "gains": gains,
                "frequencies": [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
            }, f, indent=2)
        
        self.status_label.set_text(f"Zapisano EQ do pliku: {filename}")

    def on_compare_eq(self, widget):
        """Porównaj aktualne EQ z presetem"""
        if not self.current_preset:
            self.status_label.set_text("Najpierw wybierz preset")
            return
            
        preset_gains = self.eq_preset_manager.get_preset(self.current_preset)
        current_gains = [slider.get_value() for slider, _ in self.eq_sliders]
        
        diff = [current - preset for current, preset in zip(current_gains, preset_gains)]
        
        report = f"Porównanie z presetem '{self.current_preset}':\n\n"
        for i, (freq, current, preset, diff_val) in enumerate(zip(
            [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000],
            current_gains, preset_gains, diff)):
            report += f"{freq} Hz: {current:.1f} dB (preset: {preset:.1f}, różnica: {diff_val:+.1f})\n"
        
        self.report_view.get_buffer().set_text(report)
        self.status_label.set_text("Wygenerowano porównanie EQ")

    def on_eq_changed(self, slider):
        """Callback dla zmian sliderów EQ"""
        # Znajdź indeks suwaka w liście
        index = -1
        for i, (s, db_label) in enumerate(self.eq_sliders):
            if s == slider:
                index = i
                break
        
        if index == -1:
            return  # Jeśli nie znaleziono suwaka
            
        value = slider.get_value()
        
        # Aktualizuj etykietę dB
        db_label = self.eq_sliders[index][1]
        db_label.set_text(f"{value:.1f} dB")
        
        # Zastosuj zmiany do equalizera
        self.equalizer.set_property(f"band{index}", value)
        
        # Zaktualizuj krzywą EQ do porównań
        if self.last_eq_curve:
            eq_freqs, eq_gains = self.last_eq_curve
            eq_gains[index] = value
            self.last_eq_curve = (eq_freqs, eq_gains)
            self.spectrum_widget.update_eq_curve(eq_freqs, eq_gains)

    def on_reset_eq(self, widget):
        """Resetuje ustawienia EQ"""
        for slider, db_label in self.eq_sliders:
            slider.set_value(0)
            db_label.set_text("0 dB")
            index = self.eq_sliders.index((slider, db_label))
            self.equalizer.set_property(f"band{index}", 0)
        
        self.last_eq_curve = None
        self.status_label.set_text("EQ zresetowany")

    def on_key_press(self, widget, event):
        """Obsługa klawiatury"""
        if event.keyval == Gdk.keyval_from_name('p'):
            # Reset zoomu
            self.spectrum_widget.reset_view()
            self.status_label.set_text("Zresetowano widok")
        elif event.keyval == Gdk.keyval_from_name('r'):
            # Reset przesunięcia
            self.spectrum_widget.pan_offset = 0
            self.spectrum_widget.queue_draw()
            self.status_label.set_text("Zresetowano przesunięcie")

def main():
    # Utwórz wirtualne wyjście audio
    create_virtual_sink()
    
    # Załaduj preset EQ
    eq_preset_manager = EQPresetManager()
    eq_preset_manager.load_presets()

    # Uruchom aplikację GTK
    win = AudioAnalyzerApp()
    win.show_all()
    Gtk.main()

if __name__ == "__main__":
    main()