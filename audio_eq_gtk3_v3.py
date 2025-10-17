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

class AudioNormalizer:
    """Klasa do normalizacji audio bez clippingu"""
    def __init__(self):
        self.target_lufs = -14.0  # Standard dla słuchawek (streaming)
        self.max_peak = -1.0  # Maksymalny peak (dBTP)
        self.current_lufs = -70.0
        self.current_peak = -70.0
        self.current_rms = -70.0
        self.integration_buffer = deque(maxlen=100)  # 3 sekundy przy 30Hz
        
    def calculate_lufs(self, audio_data):
        """Oblicza LUFS (Loudness Units relative to Full Scale)"""
        if len(audio_data) == 0:
            return -70.0
        # Uproszczony LUFS (bez K-weighting dla szybkości)
        rms = np.sqrt(np.mean(audio_data**2))
        if rms > 0:
            return 20 * np.log10(rms) - 0.691  # -0.691 to offset dla LUFS
        return -70.0
    
    def calculate_peak(self, audio_data):
        """Oblicza peak w dBFS"""
        if len(audio_data) == 0:
            return -70.0
        peak = np.max(np.abs(audio_data))
        if peak > 0:
            return 20 * np.log10(peak)
        return -70.0
    
    def calculate_normalization_gain(self):
        """Oblicza gain normalizacji"""
        if self.current_lufs > -70:
            lufs_gain = self.target_lufs - self.current_lufs
            # Ogranicz aby nie przekroczyć max_peak
            headroom = self.max_peak - self.current_peak
            return min(lufs_gain, headroom)
        return 0.0
    
    def update_metrics(self, audio_data):
        """Aktualizuje metryki audio"""
        self.current_lufs = self.calculate_lufs(audio_data)
        self.current_peak = self.calculate_peak(audio_data)
        rms = np.sqrt(np.mean(audio_data**2))
        self.current_rms = 20 * np.log10(rms + 1e-10)
        self.integration_buffer.append(self.current_lufs)

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
        self.dynamic_db_range = True
        self.min_db = -80
        self.max_db = 0
        self.band_history = {band: deque(maxlen=200) for band in self.freq_bands}

    def analyze_spectrum(self, spectrum_magnitude, freqs):
        """Analizuje spektrum częstotliwości - spectrum_magnitude już w dB"""
        band_analysis = {}
        
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_spectrum_db = spectrum_magnitude[mask]
            
            if len(band_spectrum_db) > 0:
                # Spektrum już w dB, nie konwertuj ponownie
                band_analysis[band_name] = {
                    'mean': np.mean(band_spectrum_db),
                    'peak': np.max(band_spectrum_db),
                    'std': np.std(band_spectrum_db),
                    'energy': np.sum(10**(band_spectrum_db/10)),  # Suma energii (liniowo)
                    'centroid': np.sum(freqs[mask] * 10**(band_spectrum_db/10)) / np.sum(10**(band_spectrum_db/10))
                }
                # Dodaj do historii
                self.band_history[band_name].append(band_analysis[band_name])
            else:
                band_analysis[band_name] = {
                    'mean': -80, 'peak': -80, 'std': 0, 'energy': 0, 'centroid': 0
                }
        
        return band_analysis

    def detect_imperfections(self, band_analysis):
        """Wykrywa niedoskonałości w poszczególnych pasmach"""
        imperfections = {}
        
        for band_name, metrics in band_analysis.items():
            issues = []
            mean_db = metrics['mean']
            peak_db = metrics['peak']
            
            # Rezonans - różnica między pikiem a średnią
            if peak_db - mean_db > self.detection_params['resonance_threshold']:
                issues.append({
                    'type': 'resonance',
                    'severity': (peak_db - mean_db) / self.detection_params['resonance_threshold'],
                    'frequency': metrics['centroid'],
                    'correction': -(peak_db - mean_db) * 0.7
                })
            
            # Null/zapadnięcie
            if mean_db < self.detection_params['null_threshold']:
                issues.append({
                    'type': 'null',
                    'severity': abs(mean_db / self.detection_params['null_threshold']),
                    'frequency': metrics['centroid'],
                    'correction': abs(mean_db) * 0.5
                })
            
            # Harshness w wysokich pasmach
            if band_name in ['mid-high', 'high']:
                if metrics['std'] > 8.0:  # Wysokie odchylenie w dB
                    issues.append({
                        'type': 'harshness',
                        'severity': metrics['std'] / 10.0,
                        'frequency': metrics['centroid'],
                        'correction': -3.0
                    })
            
            # Muddiness w niskich pasmach
            if band_name in ['low', 'mid-low']:
                # Zbyt wysoki poziom energii
                avg_energy = np.mean([10**(metrics['mean']/10)])
                if metrics['energy'] / (avg_energy + 1e-10) > 100:
                    issues.append({
                        'type': 'muddiness',
                        'severity': min(metrics['energy'] / (avg_energy * 100), 2.0),
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
                
                # Rozprzestrzeń na sąsiednie pasma
                if closest_band > 0:
                    eq_gains[closest_band - 1] += correction * weight * 0.3 * weights.get(band_name, 1.0)
                if closest_band < num_bands - 1:
                    eq_gains[closest_band + 1] += correction * weight * 0.3 * weights.get(band_name, 1.0)
        
        eq_gains = np.clip(eq_gains, -12, 12)
        return eq_freqs, eq_gains

    def update_dynamic_db_range(self, spectrum_db):
        """Aktualizuje dynamiczny zakres dB"""
        if self.dynamic_db_range and len(spectrum_db) > 0:
            valid_spectrum = spectrum_db[np.isfinite(spectrum_db)]
            if len(valid_spectrum) > 0:
                min_db = np.percentile(valid_spectrum, 5)  # 5 percentyl
                max_db = np.percentile(valid_spectrum, 95)  # 95 percentyl
                
                # Płynna aktualizacja zakresu
                self.min_db = self.min_db * 0.9 + min_db * 0.1
                self.max_db = self.max_db * 0.9 + max_db * 0.1

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
        self.connect("scroll-event", self.on_scroll)
        self.has_audio_data = False

    def on_draw(self, widget, cr):
        allocation = widget.get_allocation()
        width = allocation.width
        height = allocation.height
        
        cr.set_source_rgb(0.1, 0.1, 0.1)
        cr.rectangle(0, 0, width, height)
        cr.fill()
        
        if not self.has_audio_data:
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

    def draw_grid(self, cr, width, height):
        cr.set_source_rgba(0.3, 0.3, 0.3, 0.5)
        cr.set_line_width(1)
        
        # Poziomy dB
        db_levels = np.linspace(self.analyzer.min_db, self.analyzer.max_db, 8)
        for db in db_levels:
            y = height * (1 - (db - self.analyzer.min_db) / (self.analyzer.max_db - self.analyzer.min_db))
            cr.move_to(0, y)
            cr.line_to(width, y)
            cr.stroke()
            cr.set_source_rgba(0.6, 0.6, 0.6, 1)
            cr.move_to(5, y - 3)
            cr.show_text(f"{db:.0f} dB")
            cr.set_source_rgba(0.3, 0.3, 0.3, 0.5)
            
        # Znaczniki częstotliwości
        freq_marks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        for freq in freq_marks:
            x = width * np.log10(freq / 20) / np.log10(20000 / 20)
            if 0 <= x <= width:
                cr.move_to(x, 0)
                cr.line_to(x, height)
                cr.stroke()
                cr.set_source_rgba(0.6, 0.6, 0.6, 1)
                cr.move_to(x + 3, height - 5)
                if freq >= 1000:
                    cr.show_text(f"{freq//1000}kHz")
                else:
                    cr.show_text(f"{freq}Hz")
                cr.set_source_rgba(0.3, 0.3, 0.3, 0.5)

    def draw_spectrum(self, cr, width, height):
        if self.spectrum_data is None:
            return
        spectrum_db, freqs = self.spectrum_data
        
        cr.set_source_rgba(0.2, 0.8, 0.3, 0.7)
        cr.set_line_width(2)
        
        first_point = True
        for i in range(1, len(freqs)):
            if freqs[i] < 20 or freqs[i] > 20000:
                continue
            
            x = width * np.log10(freqs[i] / 20) / np.log10(20000 / 20)
            y = height * (1 - (spectrum_db[i] - self.analyzer.min_db) / 
                         (self.analyzer.max_db - self.analyzer.min_db))
            
            if first_point:
                cr.move_to(x, y)
                first_point = False
            else:
                cr.line_to(x, y)
        
        cr.stroke()

    def draw_eq_curve(self, cr, width, height):
        if self.eq_curve is None:
            return
        freqs, gains = self.eq_curve
        
        # Interpolacja dla płynnej krzywej
        freq_interp = np.logspace(np.log10(20), np.log10(20000), 500)
        gains_interp = np.interp(np.log10(freq_interp), np.log10(freqs), gains)
        
        cr.set_source_rgba(0.9, 0.5, 0.1, 0.8)
        cr.set_line_width(3)
        
        for i, freq in enumerate(freq_interp):
            x = width * np.log10(freq / 20) / np.log10(20000 / 20)
            # Mapuj gain -12..12 na wysokość widgetu
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
                    
                x = width * np.log10(issue['frequency'] / 20) / np.log10(20000 / 20)
                if 0 <= x <= width:
                    cr.set_source_rgba(*colors.get(issue['type'], (0.5, 0.5, 0.5, 0.5)))
                    severity = min(issue['severity'], 1.0)
                    line_height = height * 0.3 * severity
                    cr.move_to(x, 0)
                    cr.line_to(x, line_height)
                    cr.stroke()
                    
                    cr.set_source_rgba(1.0, 1.0, 1.0, 0.8)
                    cr.move_to(x + 5, line_height + 15)
                    cr.show_text(f"{issue['type']}")

    def on_scroll(self, widget, event):
        if event.direction == Gdk.ScrollDirection.UP:
            self.zoom_level *= 1.1
        elif event.direction == Gdk.ScrollDirection.DOWN:
            self.zoom_level /= 1.1
        self.zoom_level = max(0.5, min(5.0, self.zoom_level))
        self.queue_draw()
        return True

    def update_spectrum(self, spectrum_db, freqs):
        self.spectrum_data = (spectrum_db, freqs)
        self.has_audio_data = True
        self.queue_draw()

    def update_eq_curve(self, freqs, gains):
        self.eq_curve = (freqs, gains)
        self.queue_draw()

    def update_imperfections(self, imperfections):
        self.imperfections = imperfections
        self.queue_draw()

    def clear_data(self):
        self.spectrum_data = None
        self.eq_curve = None
        self.imperfections = {}
        self.has_audio_data = False
        self.queue_draw()

class BandAnalysisWidget(Gtk.DrawingArea):
    """Widget do rysowania analizy pasm w stylu DAW"""
    def __init__(self, analyzer):
        super().__init__()
        self.set_size_request(600, 300)
        self.analyzer = analyzer
        self.has_data = False
        self.connect("draw", self.on_draw)
        
    def on_draw(self, widget, cr):
        allocation = widget.get_allocation()
        width = allocation.width
        height = allocation.height
        
        # Tło
        cr.set_source_rgb(0.15, 0.15, 0.15)
        cr.rectangle(0, 0, width, height)
        cr.fill()
        
        if not self.has_data:
            cr.set_source_rgba(0.8, 0.8, 0.8, 0.5)
            cr.select_font_face("Sans", 0, 0)
            cr.set_font_size(18)
            cr.move_to(width/2 - 80, height/2)
            cr.show_text("Brak danych")
            return
        
        # Siatka
        self.draw_grid(cr, width, height)
        
        # Rysuj przebiegi dla każdego pasma
        colors = {
            'low': (1.0, 0.2, 0.2, 0.8),
            'mid-low': (1.0, 0.6, 0.2, 0.8),
            'mid': (0.2, 1.0, 0.2, 0.8),
            'mid-high': (0.2, 0.6, 1.0, 0.8),
            'high': (0.6, 0.2, 1.0, 0.8)
        }
        
        for band_name, color in colors.items():
            if band_name in self.analyzer.band_history:
                history = list(self.analyzer.band_history[band_name])
                if len(history) > 1:
                    cr.set_source_rgba(*color)
                    cr.set_line_width(2)
                    
                    for i, metrics in enumerate(history):
                        x = (i / len(history)) * width
                        mean_db = metrics['mean']
                        y = height * (1 - (mean_db - self.analyzer.min_db) / 
                                     (self.analyzer.max_db - self.analyzer.min_db))
                        
                        if i == 0:
                            cr.move_to(x, y)
                        else:
                            cr.line_to(x, y)
                    cr.stroke()
        
        # Legenda
        self.draw_legend(cr, width, height, colors)
    
    def draw_grid(self, cr, width, height):
        cr.set_source_rgba(0.3, 0.3, 0.3, 0.3)
        cr.set_line_width(1)
        
        # Poziome linie (dB)
        for i in range(5):
            y = height * i / 4
            cr.move_to(0, y)
            cr.line_to(width, y)
            cr.stroke()
            
            db = self.analyzer.max_db - (i / 4) * (self.analyzer.max_db - self.analyzer.min_db)
            cr.set_source_rgba(0.6, 0.6, 0.6, 0.8)
            cr.move_to(5, y + 15)
            cr.show_text(f"{db:.0f}dB")
            cr.set_source_rgba(0.3, 0.3, 0.3, 0.3)
    
    def draw_legend(self, cr, width, height, colors):
        legend_x = width - 150
        legend_y = 20
        
        cr.set_source_rgba(0.2, 0.2, 0.2, 0.8)
        cr.rectangle(legend_x - 10, legend_y - 10, 140, 120)
        cr.fill()
        
        for i, (band_name, color) in enumerate(colors.items()):
            y = legend_y + i * 20
            cr.set_source_rgba(*color)
            cr.rectangle(legend_x, y, 20, 10)
            cr.fill()
            
            cr.set_source_rgba(1, 1, 1, 1)
            cr.move_to(legend_x + 25, y + 10)
            cr.show_text(band_name)
    
    def update_data(self):
        self.has_data = True
        self.queue_draw()

class LevelMeterWidget(Gtk.DrawingArea):
    """Widget poziomów audio (pre/post z ostrzeżeniami)"""
    def __init__(self):
        super().__init__()
        self.set_size_request(200, 300)
        self.pre_level = -70.0
        self.post_level = -70.0
        self.pre_peak = -70.0
        self.post_peak = -70.0
        self.target_level = -14.0  # LUFS target
        self.warning_level = -1.0  # dBTP
        self.connect("draw", self.on_draw)
        
    def on_draw(self, widget, cr):
        allocation = widget.get_allocation()
        width = allocation.width
        height = allocation.height
        
        # Tło
        cr.set_source_rgb(0.1, 0.1, 0.1)
        cr.rectangle(0, 0, width, height)
        cr.fill()
        
        meter_width = width // 3
        margin = 10
        
        # Pre meter
        self.draw_meter(cr, margin, 40, meter_width, height - 80, 
                       self.pre_level, self.pre_peak, "PRE")
        
        # Post meter
        self.draw_meter(cr, width//2 - meter_width//2, 40, meter_width, height - 80,
                       self.post_level, self.post_peak, "POST")
        
        # Target indicator
        target_y = self.db_to_y(self.target_level, 40, height - 80)
        cr.set_source_rgba(1.0, 1.0, 1.0, 0.5)
        cr.set_line_width(2)
        cr.set_dash([5, 5])
        cr.move_to(0, target_y)
        cr.line_to(width, target_y)
        cr.stroke()
        cr.set_dash([])
        
        cr.move_to(width - 80, target_y - 5)
        cr.show_text(f"Target: {self.target_level:.0f} LUFS")
    
    def draw_meter(self, cr, x, y, w, h, level, peak, label):
        # Ramka
        cr.set_source_rgba(0.3, 0.3, 0.3, 1)
        cr.rectangle(x, y, w, h)
        cr.stroke()
        
        # Poziom
        level_height = self.level_to_height(level, h)
        
        # Gradient: zielony -> żółty -> czerwony
        if level > self.warning_level:
            cr.set_source_rgb(1.0, 0.0, 0.0)  # Czerwony
        elif level > -6:
            cr.set_source_rgb(1.0, 1.0, 0.0)  # Żółty
        else:
            cr.set_source_rgb(0.0, 1.0, 0.0)  # Zielony
        
        cr.rectangle(x, y + h - level_height, w, level_height)
        cr.fill()
        
        # Peak indicator
        peak_y = self.db_to_y(peak, y, h)
        cr.set_source_rgb(1.0, 1.0, 1.0)
        cr.rectangle(x, peak_y, w, 2)
        cr.fill()
        
        # Strefa ostrzeżenia
        warn_y = self.db_to_y(self.warning_level, y, h)
        cr.set_source_rgba(1.0, 0.0, 0.0, 0.3)
        cr.rectangle(x, y, w, warn_y - y)
        cr.fill()
        
        # Label
        cr.set_source_rgb(1.0, 1.0, 1.0)
        cr.move_to(x + 5, y - 5)
        cr.show_text(label)
        
        # Wartość
        cr.move_to(x + 5, y + h + 15)
        cr.show_text(f"{level:.1f} dB")
    
    def level_to_height(self, db, max_height):
        # Mapuj -70..0 dB na 0..max_height
        normalized = (db + 70) / 70
        return max(0, min(max_height, normalized * max_height))
    
    def db_to_y(self, db, y_start, height):
        normalized = (db + 70) / 70
        return y_start + height * (1 - normalized)
    
    def update_levels(self, pre, post, pre_peak, post_peak):
        self.pre_level = pre
        self.post_level = post
        self.pre_peak = pre_peak
        self.post_peak = post_peak
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
        super().__init__(title="Analizator Audio z Auto-EQ i Normalizacją")
        self.set_default_size(1400, 900)
        self.analyzer = AudioAnalyzer()
        self.normalizer = AudioNormalizer()
        self.pipeline = None
        self.is_playing = False
        self.eq_preset_manager = EQPresetManager()
        self.current_preset = None
        self.last_eq_curve = None
        self.has_audio_data = False
        self.normalize_enabled = False
        self.raw_audio_buffer = deque(maxlen=10)
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

        # Przycisk normalizacji
        self.normalize_button = Gtk.ToggleButton(label="Normalizacja OFF")
        self.normalize_button.connect("toggled", self.on_normalize_toggle)
        toolbar.pack_start(self.normalize_button, False, False, 0)

        # Menadżer presetów
        preset_label = Gtk.Label(label="Preset EQ:")
        toolbar.pack_start(preset_label, False, False, 0)
        
        self.preset_combo = Gtk.ComboBoxText()
        for preset_name in self.eq_preset_manager.presets.keys():
            self.preset_combo.append_text(preset_name)
        self.preset_combo.set_active(0)
        self.preset_combo.connect("changed", self.on_preset_changed)
        toolbar.pack_start(self.preset_combo, False, False, 0)

        self.status_label = Gtk.Label(label="Gotowy")
        toolbar.pack_start(self.status_label, False, False, 0)

        # Layout główny: poziomy podział
        main_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        main_box.pack_start(main_hbox, True, True, 0)

        # Lewa strona: notebook z analizami
        notebook = Gtk.Notebook()
        main_hbox.pack_start(notebook, True, True, 0)

        # Zakładka: Spektrum i EQ
        spectrum_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.spectrum_widget = SpectrumWidget()
        self.spectrum_widget.analyzer = self.analyzer
        spectrum_box.pack_start(self.spectrum_widget, True, True, 0)
        notebook.append_page(spectrum_box, Gtk.Label(label="Spektrum i EQ"))

        # Zakładka: Analiza pasm (nowy widget)
        self.band_analysis_widget = BandAnalysisWidget(self.analyzer)
        notebook.append_page(self.band_analysis_widget, Gtk.Label(label="Przebiegi pasm"))

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

        # Prawa strona: Level meters
        right_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        main_hbox.pack_start(right_box, False, False, 0)

        meter_frame = Gtk.Frame()
        meter_frame.set_label("Poziomy Audio")
        right_box.pack_start(meter_frame, True, True, 0)
        
        self.level_meter = LevelMeterWidget()
        meter_frame.add(self.level_meter)

        # Info o normalizacji
        norm_frame = Gtk.Frame()
        norm_frame.set_label("Info Normalizacji")
        right_box.pack_start(norm_frame, False, False, 0)
        
        norm_info_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        norm_info_box.set_margin_start(10)
        norm_info_box.set_margin_end(10)
        norm_info_box.set_margin_top(10)
        norm_info_box.set_margin_bottom(10)
        norm_frame.add(norm_info_box)
        
        self.lufs_label = Gtk.Label(label="LUFS: -- dB")
        norm_info_box.pack_start(self.lufs_label, False, False, 0)
        
        self.peak_label = Gtk.Label(label="Peak: -- dB")
        norm_info_box.pack_start(self.peak_label, False, False, 0)
        
        self.gain_label = Gtk.Label(label="Gain: 0.0 dB")
        norm_info_box.pack_start(self.gain_label, False, False, 0)

        # Kontrola EQ
        eq_frame = Gtk.Frame()
        eq_frame.set_label("Kontrola EQ (10 pasm)")
        main_box.pack_start(eq_frame, False, False, 0)
        
        eq_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        eq_frame.add(eq_box)
        
        self.eq_sliders = []
        eq_freqs = [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        
        for i, freq in enumerate(eq_freqs):
            vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            slider = Gtk.Scale(orientation=Gtk.Orientation.VERTICAL)
            slider.set_range(-12, 12)
            slider.set_value(0)
            slider.set_inverted(True)
            slider.set_size_request(40, 150)
            slider.set_draw_value(True)
            slider.connect("value-changed", self.on_eq_changed)
            
            label = Gtk.Label(label=f"{freq}Hz" if freq < 1000 else f"{freq//1000}k")
            label.set_size_request(40, -1)
            
            db_label = Gtk.Label(label="0 dB")
            db_label.set_size_request(40, -1)
            
            vbox.pack_start(slider, True, True, 0)
            vbox.pack_start(db_label, False, False, 0)
            vbox.pack_start(label, False, False, 0)
            
            eq_box.pack_start(vbox, False, False, 0)
            self.eq_sliders.append((slider, db_label))
        
        # Przyciski sterowania EQ
        eq_control_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        eq_control_box.set_margin_start(10)
        eq_box.pack_start(eq_control_box, False, False, 0)
        
        reset_button = Gtk.Button(label="Reset EQ")
        reset_button.connect("clicked", self.on_reset_eq)
        eq_control_box.pack_start(reset_button, False, False, 0)
        
        save_preset_button = Gtk.Button(label="Zapisz preset")
        save_preset_button.connect("clicked", self.on_save_preset)
        eq_control_box.pack_start(save_preset_button, False, False, 0)

    def setup_gstreamer(self):
        """Konfiguruje pipeline GStreamer"""
        self.pipeline = Gst.Pipeline.new("audio-pipeline")

        # Źródło audio
        self.src = Gst.ElementFactory.make("pulsesrc", "source")
        self.src.set_property("device", "autoeq_sink.monitor")
        
        # Konwersja
        self.convert1 = Gst.ElementFactory.make("audioconvert", "convert1")
        self.resample = Gst.ElementFactory.make("audioresample", "resample")
        
        # Capsfilter dla stałego formatu
        self.caps = Gst.ElementFactory.make("capsfilter", "caps")
        caps_struct = Gst.Caps.from_string("audio/x-raw,format=F32LE,rate=44100,channels=2")
        self.caps.set_property("caps", caps_struct)
        
        # Equalizer
        self.equalizer = Gst.ElementFactory.make("equalizer-10bands", "eq")
        
        # Volume (dla normalizacji)
        self.volume = Gst.ElementFactory.make("volume", "volume")
        self.volume.set_property("volume", 1.0)
        
        # Spectrum analyzer
        self.spectrum = Gst.ElementFactory.make("spectrum", "spectrum")
        self.spectrum.set_property("bands", 2048)
        self.spectrum.set_property("threshold", -100)
        self.spectrum.set_property("interval", 50000000)  # 50ms
        self.spectrum.set_property("post-messages", True)
        self.spectrum.set_property("message-magnitude", True)
        
        # Level meter
        self.level = Gst.ElementFactory.make("level", "level")
        self.level.set_property("post-messages", True)
        self.level.set_property("interval", 50000000)
        
        # Wyjście
        self.convert2 = Gst.ElementFactory.make("audioconvert", "convert2")
        self.sink = Gst.ElementFactory.make("autoaudiosink", "sink")

        # Dodaj elementy
        elements = [self.src, self.convert1, self.resample, self.caps, 
                   self.equalizer, self.volume, self.spectrum, self.level, 
                   self.convert2, self.sink]
        
        for element in elements:
            if element:
                self.pipeline.add(element)
            else:
                print(f"Błąd: nie można utworzyć elementu")
                return

        # Łączenie
        if not self.src.link(self.convert1):
            print("Błąd łączenia src->convert1")
        if not self.convert1.link(self.resample):
            print("Błąd łączenia convert1->resample")
        if not self.resample.link(self.caps):
            print("Błąd łączenia resample->caps")
        if not self.caps.link(self.equalizer):
            print("Błąd łączenia caps->equalizer")
        if not self.equalizer.link(self.volume):
            print("Błąd łączenia equalizer->volume")
        if not self.volume.link(self.spectrum):
            print("Błąd łączenia volume->spectrum")
        if not self.spectrum.link(self.level):
            print("Błąd łączenia spectrum->level")
        if not self.level.link(self.convert2):
            print("Błąd łączenia level->convert2")
        if not self.convert2.link(self.sink):
            print("Błąd łączenia convert2->sink")

        # Bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_bus_message)

    def on_bus_message(self, bus, message):
        """Obsługa wiadomości z GStreamer"""
        if message.type == Gst.MessageType.ELEMENT:
            struct = message.get_structure()
            
            if struct and struct.get_name() == "spectrum":
                # Pobierz magnitude (już w dB!)
                magnitudes = struct.get_value("magnitude")
                if magnitudes:
                    spectrum_db = np.array(magnitudes)
                    num_bands = len(spectrum_db)
                    freqs = np.linspace(0, self.analyzer.sample_rate / 2, num_bands)
                    
                    # Filtruj nieprawidłowe wartości
                    valid_mask = np.isfinite(spectrum_db)
                    spectrum_db = spectrum_db[valid_mask]
                    freqs = freqs[valid_mask]
                    
                    if len(spectrum_db) > 0:
                        self.has_audio_data = True
                        
                        # Aktualizuj zakres dynamiczny
                        self.analyzer.update_dynamic_db_range(spectrum_db)
                        
                        # Analiza pasm
                        band_analysis = self.analyzer.analyze_spectrum(spectrum_db, freqs)
                        
                        # Wykryj niedoskonałości
                        imperfections = self.analyzer.detect_imperfections(band_analysis)
                        
                        # Generuj krzywą EQ
                        eq_freqs, eq_gains = self.analyzer.generate_eq_curve(imperfections)
                        
                        # Aktualizuj widgety
                        GLib.idle_add(self.spectrum_widget.update_spectrum, spectrum_db, freqs)
                        GLib.idle_add(self.spectrum_widget.update_eq_curve, eq_freqs, eq_gains)
                        GLib.idle_add(self.spectrum_widget.update_imperfections, imperfections)
                        GLib.idle_add(self.band_analysis_widget.update_data)
            
            elif struct and struct.get_name() == "level":
                # Pobierz poziomy RMS i peak
                rms_values = struct.get_value("rms")
                peak_values = struct.get_value("peak")
                
                if rms_values and peak_values:
                    # Średnia z kanałów
                    rms_db = np.mean(rms_values)
                    peak_db = np.mean(peak_values)
                    
                    # Symulacja pre-level (przed EQ i normalizacją)
                    # W prawdziwej implementacji potrzebny byłby drugi element level przed EQ
                    pre_level = rms_db
                    pre_peak = peak_db
                    
                    # Post-level (po EQ i normalizacji)
                    if self.normalize_enabled:
                        gain = self.normalizer.calculate_normalization_gain()
                        post_level = rms_db + gain
                        post_peak = peak_db + gain
                    else:
                        post_level = rms_db
                        post_peak = peak_db
                    
                    # Aktualizuj normalizer
                    # Utworzenie dummy audio data dla LUFS
                    dummy_audio = np.random.randn(4096) * 10**(rms_db/20)
                    self.normalizer.update_metrics(dummy_audio)
                    
                    # Aktualizuj widgety
                    GLib.idle_add(self.level_meter.update_levels, pre_level, post_level, 
                                 pre_peak, post_peak)
                    GLib.idle_add(self.update_norm_info)

    def update_norm_info(self):
        """Aktualizuj informacje o normalizacji"""
        self.lufs_label.set_text(f"LUFS: {self.normalizer.current_lufs:.1f} dB")
        self.peak_label.set_text(f"Peak: {self.normalizer.current_peak:.1f} dBTP")
        
        if self.normalize_enabled:
            gain = self.normalizer.calculate_normalization_gain()
            self.gain_label.set_text(f"Gain: {gain:+.1f} dB")
        else:
            self.gain_label.set_text("Gain: 0.0 dB (OFF)")

    def on_normalize_toggle(self, button):
        """Przełącz normalizację"""
        self.normalize_enabled = button.get_active()
        
        if self.normalize_enabled:
            button.set_label("Normalizacja ON")
            self.status_label.set_text("Normalizacja włączona")
            # Zastosuj gain
            GLib.timeout_add(50, self.apply_normalization_gain)
        else:
            button.set_label("Normalizacja OFF")
            self.volume.set_property("volume", 1.0)
            self.status_label.set_text("Normalizacja wyłączona")

    def apply_normalization_gain(self):
        """Zastosuj gain normalizacji"""
        if self.normalize_enabled and self.is_playing:
            gain_db = self.normalizer.calculate_normalization_gain()
            gain_linear = 10 ** (gain_db / 20)
            # Ogranicz gain dla bezpieczeństwa
            gain_linear = max(0.1, min(3.0, gain_linear))
            self.volume.set_property("volume", gain_linear)
            return True  # Kontynuuj timeout
        return False  # Zatrzymaj timeout

    def on_play_pause(self, widget):
        """Rozpoczyna/zatrzymuje nasłuchiwanie"""
        if not self.is_playing:
            self.pipeline.set_state(Gst.State.PLAYING)
            self.play_button.set_label("Zatrzymaj")
            self.is_playing = True
            self.status_label.set_text("Nasłuchiwanie...")
            self.has_audio_data = False
        else:
            self.pipeline.set_state(Gst.State.PAUSED)
            self.play_button.set_label("Nasłuchuj")
            self.is_playing = False
            self.status_label.set_text("Zatrzymano")
            self.has_audio_data = False
            self.clear_spectrum_view()

    def clear_spectrum_view(self):
        """Czyści widok spektrum i EQ"""
        self.spectrum_widget.clear_data()
        self.band_analysis_widget.has_data = False
        self.band_analysis_widget.queue_draw()
        
        for slider, db_label in self.eq_sliders:
            GLib.idle_add(slider.set_value, 0)
            GLib.idle_add(db_label.set_text, "0 dB")
        
        self.report_view.get_buffer().set_text("Brak danych audio. Rozpocznij odtwarzanie.")

    def on_analyze(self, widget):
        """Przeprowadza analizę bieżącego spektrum"""
        if not self.has_audio_data:
            self.status_label.set_text("Brak danych audio - rozpocznij odtwarzanie")
            return
        
        # Analiza z historii pasm
        report = "=== RAPORT ANALIZY AUDIO ===\n\n"
        
        for band_name, history in self.analyzer.band_history.items():
            if len(history) > 0:
                recent_metrics = list(history)[-10:]  # Ostatnie 10 pomiarów
                avg_mean = np.mean([m['mean'] for m in recent_metrics])
                avg_peak = np.mean([m['peak'] for m in recent_metrics])
                avg_std = np.mean([m['std'] for m in recent_metrics])
                
                report += f"\n{band_name.upper()}:\n"
                report += f"  Średnia: {avg_mean:.1f} dB\n"
                report += f"  Szczyt: {avg_peak:.1f} dB\n"
                report += f"  Odchylenie: {avg_std:.1f} dB\n"
        
        report += f"\n=== METRYKI NORMALIZACJI ===\n"
        report += f"LUFS: {self.normalizer.current_lufs:.1f} dB\n"
        report += f"Peak: {self.normalizer.current_peak:.1f} dBTP\n"
        report += f"Target: {self.normalizer.target_lufs:.1f} LUFS\n"
        
        if self.normalize_enabled:
            gain = self.normalizer.calculate_normalization_gain()
            report += f"Zastosowany gain: {gain:+.1f} dB\n"
        
        self.report_view.get_buffer().set_text(report)
        self.status_label.set_text("Analiza zakończona")

    def on_apply_eq(self, widget):
        """Zastosuj automatyczną krzywą EQ"""
        if not self.has_audio_data:
            self.status_label.set_text("Brak danych - rozpocznij odtwarzanie")
            return
        
        # Pobierz ostatnie dane z historii
        if len(self.analyzer.band_history['low']) == 0:
            self.status_label.set_text("Brak wystarczających danych")
            return
        
        # Utwórz band_analysis z historii
        band_analysis = {}
        for band_name in self.analyzer.freq_bands.keys():
            if len(self.analyzer.band_history[band_name]) > 0:
                recent = list(self.analyzer.band_history[band_name])[-1]
                band_analysis[band_name] = recent
        
        imperfections = self.analyzer.detect_imperfections(band_analysis)
        weights = self.band_weight_widget.get_weights()
        eq_freqs, eq_gains = self.analyzer.generate_eq_curve(imperfections, weights)
        
        for i, (slider, db_label) in enumerate(self.eq_sliders):
            gain = eq_gains[i]
            slider.set_value(gain)
            db_label.set_text(f"{gain:.1f} dB")
            self.equalizer.set_property(f"band{i}", gain)
        
        self.status_label.set_text("Zastosowano Auto-EQ")

    def on_preset_changed(self, combo):
        """Zastosuj wybrany preset EQ"""
        preset_name = combo.get_active_text()
        if preset_name:
            gains = self.eq_preset_manager.get_preset(preset_name)
            self.current_preset = preset_name
            
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

    def on_eq_changed(self, slider):
        """Callback dla zmian sliderów EQ"""
        index = -1
        for i, (s, db_label) in enumerate(self.eq_sliders):
            if s == slider:
                index = i
                break
        
        if index == -1:
            return
            
        value = slider.get_value()
        db_label = self.eq_sliders[index][1]
        db_label.set_text(f"{value:.1f} dB")
        self.equalizer.set_property(f"band{index}", value)

    def on_reset_eq(self, widget):
        """Resetuje ustawienia EQ"""
        for i, (slider, db_label) in enumerate(self.eq_sliders):
            slider.set_value(0)
            db_label.set_text("0 dB")
            self.equalizer.set_property(f"band{i}", 0)
        
        self.status_label.set_text("EQ zresetowany")

    def on_key_press(self, widget, event):
        """Obsługa klawiatury"""
        if event.keyval == Gdk.keyval_from_name('p'):
            self.spectrum_widget.zoom_level = 1.0
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
            