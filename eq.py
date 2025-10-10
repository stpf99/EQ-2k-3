#!/usr/bin/env python3
"""
Zaawansowana aplikacja do analizy audio z automatyczną korekcją EQ
Wersja na GTK4 i Wayland.
Tworzy wirtualne wyjście audio dla dowolnej aplikacji, nasłuchuje dźwięk,
przetwarza go i wysyła do domyślnego wyjścia.
"""
import gi
import subprocess
import numpy as np
import math
import os
from collections import deque
from scipy import signal
from scipy.fft import rfft, rfftfreq

# Wymagamy wersji 4.0 dla GTK
gi.require_version('Gtk', '4.0')
gi.require_version('Gst', '1.0')
gi.require_version('Gdk', '4.0')
gi.require_version('Graphene', '1.0')

from gi.repository import Gtk, Gst, GLib, Gdk, Graphene

# Inicjalizacja GStreamer
Gst.init(None)

def create_virtual_sink(sink_name="autoeq_sink"):
    """Tworzy wirtualne wyjście audio za pomocą pactl."""
    # Sprawdź, czy nie istnieje
    result = subprocess.run(["pactl", "list", "sinks", "short"], capture_output=True, text=True)
    if sink_name in result.stdout:
        print(f"Wirtualne wyjście '{sink_name}' już istnieje.")
        return

    try:
        subprocess.run(
            ["pactl", "load-module", "module-null-sink", f"sink_name={sink_name}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"Utworzono wirtualne wyjście audio: {sink_name}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Nie udało się utworzyć wirtualnego wyjścia: {e.stderr.decode('utf-8') if e.stderr else e}")
        print("Upewnij się, że pulseaudio jest zainstalowane i uruchomione.")

class AudioAnalyzer:
    """Klasa analizująca audio i wykrywająca niedoskonałości (bez zmian)"""
    def __init__(self):
        self.sample_rate = 44100
        self.buffer_size = 4096
        self.freq_bands = {
            'low': (20, 250), 'mid-low': (250, 800), 'mid': (800, 3000),
            'mid-high': (3000, 8000), 'high': (8000, 20000)
        }
        self.time_analysis = {band: deque(maxlen=100) for band in self.freq_bands}
        self.imperfections = {band: [] for band in self.freq_bands}
        self.detection_params = {
            'resonance_threshold': 6.0, 'null_threshold': -12.0,
            'harshness_factor': 1.5, 'muddiness_threshold': 0.7
        }

    def analyze_spectrum(self, audio_data):
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
                    'mean': np.mean(band_spectrum), 'peak': np.max(band_spectrum),
                    'std': np.std(band_spectrum), 'energy': np.sum(band_spectrum**2),
                    'centroid': np.sum(freqs[mask] * band_spectrum) / np.sum(band_spectrum)
                }
            else:
                band_analysis[band_name] = {'mean': 0, 'peak': 0, 'std': 0, 'energy': 0, 'centroid': 0}
        return band_analysis, spectrum, freqs

    def detect_imperfections(self, band_analysis):
        imperfections = {}
        for band_name, metrics in band_analysis.items():
            issues = []
            mean_db = 20 * np.log10(metrics['mean'] + 1e-10)
            peak_db = 20 * np.log10(metrics['peak'] + 1e-10)
            if peak_db - mean_db > self.detection_params['resonance_threshold']:
                issues.append({'type': 'resonance', 'severity': (peak_db - mean_db) / self.detection_params['resonance_threshold'], 'frequency': metrics['centroid'], 'correction': -(peak_db - mean_db) * 0.7})
            if mean_db < self.detection_params['null_threshold']:
                issues.append({'type': 'null', 'severity': abs(mean_db / self.detection_params['null_threshold']), 'frequency': metrics['centroid'], 'correction': abs(mean_db) * 0.5})
            if band_name in ['mid-high', 'high']:
                if metrics['std'] / (metrics['mean'] + 1e-10) > self.detection_params['harshness_factor']:
                    issues.append({'type': 'harshness', 'severity': metrics['std'] / (metrics['mean'] + 1e-10), 'frequency': metrics['centroid'], 'correction': -3.0})
            if band_name in ['low', 'mid-low']:
                if metrics['energy'] / (metrics['mean'] + 1e-10) > self.detection_params['muddiness_threshold']:
                    issues.append({'type': 'muddiness', 'severity': metrics['energy'] / (metrics['mean'] + 1e-10), 'frequency': metrics['centroid'], 'correction': -2.0})
            imperfections[band_name] = issues
        return imperfections

    def generate_eq_curve(self, imperfections, weights=None, num_bands=10):
        eq_freqs = np.array([31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
        eq_gains = np.zeros(num_bands)
        if weights is None: weights = {band: 1.0 for band in self.freq_bands}
        for band_name, issues in imperfections.items():
            for issue in issues:
                freq = issue['frequency']; correction = issue['correction']
                closest_band = np.argmin(np.abs(eq_freqs - freq))
                weight = 1.0 / (1.0 + np.abs(eq_freqs[closest_band] - freq) / 1000)
                eq_gains[closest_band] += correction * weight * issue['severity'] * weights.get(band_name, 1.0)
                if closest_band > 0: eq_gains[closest_band - 1] += correction * weight * 0.3 * weights.get(band_name, 1.0)
                if closest_band < num_bands - 1: eq_gains[closest_band + 1] += correction * weight * 0.3 * weights.get(band_name, 1.0)
        eq_gains = np.clip(eq_gains, -12, 12)
        return eq_freqs, eq_gains

class SpectrumWidget(Gtk.DrawingArea):
    """Widget do rysowania spektrum i krzywych EQ w GTK4"""
    def __init__(self):
        super().__init__()
        self.set_draw_func(self.on_draw) # GTK4: Używamy set_draw_func
        self.set_size_request(800, 400)
        self.spectrum_data = None
        self.eq_curve = None
        self.imperfections = {}

    def on_draw(self, area, cr, width, height):
        """GTK4: Funkcja rysująca ma nową sygnaturę."""
        # Tło
        cr.set_source_rgb(0.1, 0.1, 0.1)
        cr.rectangle(0, 0, width, height)
        cr.fill()

        self.draw_grid(cr, width, height)
        if self.spectrum_data: self.draw_spectrum(cr, width, height)
        if self.eq_curve: self.draw_eq_curve(cr, width, height)
        self.draw_imperfections(cr, width, height)

    def draw_grid(self, cr, width, height):
        cr.set_source_rgba(0.3, 0.3, 0.3, 0.5); cr.set_line_width(1)
        db_levels = [-40, -30, -20, -10, 0, 10]
        for db in db_levels:
            y = height * (1 - (db + 40) / 50)
            cr.move_to(0, y); cr.line_to(width, y); cr.stroke()
            cr.set_source_rgba(0.6, 0.6, 0.6, 1)
            cr.move_to(5, y - 3); cr.show_text(f"{db} dB")
        freq_marks = [100, 1000, 10000]
        for freq in freq_marks:
            x = width * np.log10(freq / 20) / np.log10(20000 / 20)
            cr.set_source_rgba(0.3, 0.3, 0.3, 0.5)
            cr.move_to(x, 0); cr.line_to(x, height); cr.stroke()
            cr.set_source_rgba(0.6, 0.6, 0.6, 1)
            cr.move_to(x + 3, height - 5); cr.show_text(f"{freq} Hz")

    def draw_spectrum(self, cr, width, height):
        spectrum, freqs = self.spectrum_data
        spectrum_db = 20 * np.log10(np.abs(spectrum) + 1e-10)
        spectrum_db = np.clip(spectrum_db, -40, 10)
        cr.set_source_rgba(0.2, 0.8, 0.3, 0.7); cr.set_line_width(2)
        for i in range(1, len(freqs)):
            if not (20 <= freqs[i] <= 20000): continue
            x1 = width * np.log10(freqs[i-1] / 20) / np.log10(20000 / 20)
            x2 = width * np.log10(freqs[i] / 20) / np.log10(20000 / 20)
            y1 = height * (1 - (spectrum_db[i-1] + 40) / 50)
            y2 = height * (1 - (spectrum_db[i] + 40) / 50)
            if i == 1: cr.move_to(x1, y1)
            cr.line_to(x2, y2)
        cr.stroke()

    def draw_eq_curve(self, cr, width, height):
        freqs, gains = self.eq_curve
        freq_interp = np.logspace(np.log10(20), np.log10(20000), 500)
        gains_interp = np.interp(np.log10(freq_interp), np.log10(freqs), gains)
        cr.set_source_rgba(0.9, 0.5, 0.1, 0.8); cr.set_line_width(3)
        for i in range(len(freq_interp)):
            x = width * np.log10(freq_interp[i] / 20) / np.log10(20000 / 20)
            y = height * (0.5 - gains_interp[i] / 24)
            if i == 0: cr.move_to(x, y)
            else: cr.line_to(x, y)
        cr.stroke()

    def draw_imperfections(self, cr, width, height):
        colors = {'resonance': (1.0, 0.2, 0.2, 0.6), 'null': (0.2, 0.2, 1.0, 0.6), 'harshness': (1.0, 1.0, 0.2, 0.6), 'muddiness': (0.6, 0.3, 0.1, 0.6)}
        for band_name, issues in self.imperfections.items():
            for issue in issues:
                if not (20 <= issue['frequency'] <= 20000): continue
                x = width * np.log10(issue['frequency'] / 20) / np.log10(20000 / 20)
                cr.set_source_rgba(*colors.get(issue['type'], (0.5, 0.5, 0.5, 0.5)))
                cr.arc(x, height * 0.1, 5 * issue['severity'], 0, 2 * math.pi)
                cr.fill()

    def update_data(self, spectrum=None, freqs=None, eq_curve=None, imperfections=None):
        if spectrum is not None: self.spectrum_data = (spectrum, freqs)
        if eq_curve is not None: self.eq_curve = eq_curve
        if imperfections is not None: self.imperfections = imperfections
        self.queue_draw() # GTK4: queue_draw() odświeża widżet

class BandWeightWidget(Gtk.Box):
    """Widget do ustawiania wag dla poszczególnych pasm (bez zmian)"""
    def __init__(self, analyzer):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.analyzer = analyzer; self.sliders = {}
        for band_name in self.analyzer.freq_bands.keys():
            frame = Gtk.Frame(label=f"Pasmo: {band_name}") # Gtk.Frame jest nadal dostępne
            hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
            label = Gtk.Label(label=f"Waga: ")
            slider = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=Gtk.Adjustment.new(1, 0, 2, 0.1, 0.1, 0))
            slider.set_draw_value(True)
            slider.connect("value-changed", self.on_weight_changed, band_name)
            hbox.append(label); hbox.append(slider) # GTK4: .append() zamiast .pack_start()
            frame.set_child(hbox) # GTK4: .set_child() zamiast .add()
            self.append(frame); self.sliders[band_name] = slider

    def on_weight_changed(self, slider, band_name): print(f"Zmieniono wagę dla pasma {band_name} na {slider.get_value()}")
    def get_weights(self): return {band_name: slider.get_value() for band_name, slider in self.sliders.items()}

class AudioAnalyzerApp(Gtk.ApplicationWindow):
    """Główne okno aplikacji w GTK4"""
    def __init__(self, application):
        super().__init__(application=application)
        self.set_title("Analizator Audio z Auto-EQ (GTK4 + Wayland)")
        self.set_default_size(1200, 800)
        self.analyzer = AudioAnalyzer()
        self.pipeline = None; self.is_playing = False
        self.band_analysis = {}
        self.setup_ui()
        self.setup_gstreamer()

    def setup_ui(self):
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        main_box.set_margin_top(10); main_box.set_margin_bottom(10)
        main_box.set_margin_start(10); main_box.set_margin_end(10)
        self.set_child(main_box) # GTK4: .set_child() zamiast .add()

        toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        main_box.append(toolbar)

        self.play_button = Gtk.Button(label="Nasłuchuj")
        self.play_button.connect("clicked", self.on_play_pause)
        toolbar.append(self.play_button)

        self.analyze_button = Gtk.Button(label="Analizuj")
        self.analyze_button.connect("clicked", self.on_analyze)
        toolbar.append(self.analyze_button)

        self.apply_eq_button = Gtk.Button(label="Zastosuj Auto-EQ")
        self.apply_eq_button.connect("clicked", self.on_apply_eq)
        toolbar.append(self.apply_eq_button)

        self.status_label = Gtk.Label(label="Gotowy")
        toolbar.append(self.status_label)

        notebook = Gtk.Notebook()
        main_box.append(notebook)

        # --- Strona Spektrum ---
        spectrum_page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.spectrum_widget = SpectrumWidget()
        spectrum_page.append(self.spectrum_widget)
        notebook.append_page(spectrum_page, Gtk.Label(label="Spektrum i EQ"))

        # --- Strona wag ---
        self.band_weight_widget = BandWeightWidget(self.analyzer)
        notebook.append_page(self.band_weight_widget, Gtk.Label(label="Wagi pasm"))

        # --- Strona Raportu ---
        self.report_view = Gtk.TextView(editable=False, wrap_mode=Gtk.WrapMode.WORD)
        report_scroll = Gtk.ScrolledWindow()
        report_scroll.set_child(self.report_view)
        notebook.append_page(report_scroll, Gtk.Label(label="Raport"))

        # --- Kontrola EQ ---
        eq_frame = Gtk.Frame(label="Kontrola EQ (10 pasm)")
        main_box.append(eq_frame)
        eq_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        eq_frame.set_child(eq_box)
        self.eq_sliders = []
        eq_freqs = [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        for freq in eq_freqs:
            vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            slider = Gtk.Scale(orientation=Gtk.Orientation.VERTICAL, adjustment=Gtk.Adjustment.new(0, -12, 12, 1, 1, 0))
            slider.set_inverted(True); slider.set_size_request(40, 150); slider.set_draw_value(True)
            slider.connect("value-changed", self.on_eq_changed)
            label = Gtk.Label(label=f"{freq}Hz"); label.set_size_request(40, -1)
            vbox.append(slider); vbox.append(label)
            eq_box.append(vbox)
            self.eq_sliders.append(slider)
        reset_button = Gtk.Button(label="Reset EQ")
        reset_button.connect("clicked", self.on_reset_eq)
        eq_box.append(reset_button)

    def setup_gstreamer(self):
        """Konfiguracja pipeline GStreamer (z poprawkami z poprzedniej odpowiedzi)"""
        self.pipeline = Gst.Pipeline.new("audio-pipeline")
        self.src = Gst.ElementFactory.make("pulsesrc", "source")
        self.convert = Gst.ElementFactory.make("audioconvert", "convert")
        self.equalizer = Gst.ElementFactory.make("equalizer-10bands", "eq")
        self.spectrum = Gst.ElementFactory.make("spectrum", "spectrum")
        self.sink = Gst.ElementFactory.make("autoaudiosink", "sink")

        if not all([self.src, self.convert, self.equalizer, self.spectrum, self.sink]):
            print("BŁĄD: Nie udało się utworzyć jednego z elementów GStreamer.")
            self.status_label.set_text("Błąd: Brak elementu GStreamer")
            return

        self.src.set_property("device", "autoeq_sink.monitor")
        self.spectrum.set_property("bands", 1024); self.spectrum.set_property("threshold", -80); self.spectrum.set_property("interval", 50000000)
        self.pipeline.add(self.src, self.convert, self.equalizer, self.spectrum, self.sink)
        
        if not (self.src.link(self.convert) and self.convert.link(self.equalizer) and self.equalizer.link(self.spectrum) and self.spectrum.link(self.sink)):
            print("BŁĄD: Nie udało się połączyć elementów pipeline.")
            self.status_label.set_text("Błąd połączenia GStreamer")
            return

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_bus_message)
        print("Pipeline GStreamer skonfigurowany pomyślnie.")

    def on_bus_message(self, bus, message):
        """Obsługa wiadomości z GStreamer - bezpieczna dla wątków"""
        if message.type == Gst.MessageType.ELEMENT:
            struct = message.get_structure()
            if struct.get_name() == "spectrum":
                magnitudes = struct.get_value("magnitude")
                # Przenieś przetwarzanie do głównego wątku GTK
                GLib.idle_add(self.process_spectrum_data, magnitudes)

    def process_spectrum_data(self, magnitudes):
        """Przetwarza dane i aktualizuje UI w głównym wątku GTK."""
        spectrum = np.array(magnitudes, dtype=np.float32)
        freqs = np.linspace(0, self.analyzer.sample_rate / 2, len(spectrum))
        band_analysis, _, _ = self.analyzer.analyze_spectrum(spectrum)
        self.band_analysis = band_analysis
        imperfections = self.analyzer.detect_imperfections(band_analysis)
        
        eq_freqs, eq_gains = self.analyzer.generate_eq_curve(imperfections)
        for i, gain in enumerate(eq_gains):
            self.eq_sliders[i].set_value(gain)
            self.equalizer.set_property(f"band{i}", gain)
        
        self.spectrum_widget.update_data(spectrum=spectrum, freqs=freqs, eq_curve=(eq_freqs, eq_gains), imperfections=imperfections)
        return False # Ważne dla GLib.idle_add

    def on_play_pause(self, widget):
        if not self.is_playing:
            self.pipeline.set_state(Gst.State.PLAYING)
            self.play_button.set_label("Zatrzymaj"); self.is_playing = True
        else:
            self.pipeline.set_state(Gst.State.PAUSED)
            self.play_button.set_label("Nasłuchuj"); self.is_playing = False

    def on_analyze(self, widget):
        if not self.band_analysis: self.status_label.set_text("Brak danych do analizy"); return
        self.status_label.set_text("Analizowanie...")
        imperfections = self.analyzer.detect_imperfections(self.band_analysis)
        eq_freqs, eq_gains = self.analyzer.generate_eq_curve(imperfections)
        report = "=== RAPORT ANALIZY AUDIO ===\n\n"
        for band_name, issues in imperfections.items():
            if issues:
                report += f"\n{band_name.upper()}:\n"
                for issue in issues:
                    report += f"  - {issue['type']}: częstotliwość {issue['frequency']:.0f} Hz, ważność: {issue['severity']:.2f}, korekcja: {issue['correction']:.1f} dB\n"
        report += "\n=== SUGEROWANE USTAWIENIA EQ ===\n"
        for freq, gain in zip(eq_freqs, eq_gains): report += f"{freq} Hz: {gain:.1f} dB\n"
        self.report_view.get_buffer().set_text(report)
        self.status_label.set_text("Analiza zakończona")

    def on_apply_eq(self, widget):
        if hasattr(self, 'band_analysis'):
            imperfections = self.analyzer.detect_imperfections(self.band_analysis)
            weights = self.band_weight_widget.get_weights()
            eq_freqs, eq_gains = self.analyzer.generate_eq_curve(imperfections, weights)
            for i, gain in enumerate(eq_gains):
                self.eq_sliders[i].set_value(gain)
                self.equalizer.set_property(f"band{i}", gain)
            self.status_label.set_text("Zastosowano Auto-EQ z wagami pasm")

    def on_eq_changed(self, slider):
        index = self.eq_sliders.index(slider); value = slider.get_value()
        self.equalizer.set_property(f"band{index}", value)

    def on_reset_eq(self, widget):
        for i, slider in enumerate(self.eq_sliders):
            slider.set_value(0); self.equalizer.set_property(f"band{i}", 0)
        self.status_label.set_text("EQ zresetowany")

def on_activate(app):
    """Funkcja wywoływana przy aktywacji aplikacji GTK4"""
    win = AudioAnalyzerApp(application=app)
    win.present()

def main():
    create_virtual_sink()
    app = Gtk.Application(application_id="org.example.audioanalyzer.gtk4")
    app.connect("activate", on_activate)
    app.run()

if __name__ == "__main__":
    main()