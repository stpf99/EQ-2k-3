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
from collections import deque
from scipy import signal
from scipy.fft import rfft, rfftfreq

gi.require_version('Gtk', '4.0')
gi.require_version('Gst', '1.0')
from gi.repository import Gtk, Gst, GLib

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
                if metrics['std'] / metrics['mean'] > self.detection_params['harshness_factor']:
                    issues.append({
                        'type': 'harshness',
                        'severity': metrics['std'] / metrics['mean'],
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

class SpectrumWidget(Gtk.DrawingArea):
    """Widget do rysowania spektrum i krzywych EQ"""
    def __init__(self):
        super().__init__()
        self.set_size_request(800, 400)
        self.spectrum_data = None
        self.eq_curve = None
        self.imperfections = {}
        self.set_draw_func(self.draw)

    def draw(self, area, cr, width, height):
        cr.set_source_rgb(0.1, 0.1, 0.1)
        cr.rectangle(0, 0, width, height)
        cr.fill()
        self.draw_grid(cr, width, height)
        if self.spectrum_data is not None:
            self.draw_spectrum(cr, width, height)
        if self.eq_curve is not None:
            self.draw_eq_curve(cr, width, height)
        self.draw_imperfections(cr, width, height)

    def draw_grid(self, cr, width, height):
        cr.set_source_rgba(0.3, 0.3, 0.3, 0.5)
        cr.set_line_width(1)
        db_levels = [-40, -30, -20, -10, 0, 10]
        for db in db_levels:
            y = height * (1 - (db + 40) / 50)
            cr.move_to(0, y)
            cr.line_to(width, y)
            cr.stroke()
            cr.set_source_rgba(0.6, 0.6, 0.6, 1)
            cr.move_to(5, y - 3)
            cr.show_text(f"{db} dB")
        freq_marks = [100, 1000, 10000]
        for freq in freq_marks:
            x = width * np.log10(freq / 20) / np.log10(20000 / 20)
            cr.set_source_rgba(0.3, 0.3, 0.3, 0.5)
            cr.move_to(x, 0)
            cr.line_to(x, height)
            cr.stroke()
            cr.set_source_rgba(0.6, 0.6, 0.6, 1)
            cr.move_to(x + 3, height - 5)
            cr.show_text(f"{freq} Hz")

    def draw_spectrum(self, cr, width, height):
        if self.spectrum_data is None:
            return
        spectrum, freqs = self.spectrum_data
        spectrum = np.abs(spectrum)
        spectrum_db = 20 * np.log10(spectrum + 1e-10)
        spectrum_db = np.clip(spectrum_db, -40, 10)
        cr.set_source_rgba(0.2, 0.8, 0.3, 0.7)
        cr.set_line_width(2)
        for i in range(1, len(freqs)):
            if freqs[i] < 20 or freqs[i] > 20000:
                continue
            if freqs[i-1] <= 0 or freqs[i] <= 0:
                continue
            x1 = width * np.log10(freqs[i-1] / 20) / np.log10(20000 / 20)
            x2 = width * np.log10(freqs[i] / 20) / np.log10(20000 / 20)
            y1 = height * (1 - (spectrum_db[i-1] + 40) / 50)
            y2 = height * (1 - (spectrum_db[i] + 40) / 50)
            if i == 1:
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
            x = width * np.log10(freq_interp[i] / 20) / np.log10(20000 / 20)
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
                cr.set_source_rgba(*colors.get(issue['type'], (0.5, 0.5, 0.5, 0.5)))
                cr.arc(x, height * 0.1, 5 * issue['severity'], 0, 2 * math.pi)
                cr.fill()

    def update_spectrum(self, spectrum, freqs):
        self.spectrum_data = (spectrum, freqs)
        self.queue_draw()

    def update_eq_curve(self, freqs, gains):
        self.eq_curve = (freqs, gains)
        self.queue_draw()

    def update_imperfections(self, imperfections):
        self.imperfections = imperfections
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
            hbox.append(label)
            slider = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
            slider.set_range(0, 2)
            slider.set_value(1)
            slider.set_draw_value(True)
            slider.connect("value-changed", self.on_weight_changed, band_name)
            hbox.append(slider)
            frame.set_child(hbox)
            self.append(frame)
            self.sliders[band_name] = slider

    def on_weight_changed(self, slider, band_name):
        value = slider.get_value()
        print(f"Zmieniono wagę dla pasma {band_name} na {value}")

    def get_weights(self):
        weights = {}
        for band_name, slider in self.sliders.items():
            weights[band_name] = slider.get_value()
        return weights

class AudioAnalyzerApp(Gtk.ApplicationWindow):
    """Główne okno aplikacji"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_title("Analizator Audio z Auto-EQ (Wirtualne Wyjście)")
        self.set_default_size(1200, 800)
        self.analyzer = AudioAnalyzer()
        self.pipeline = None
        self.is_playing = False
        self.setup_ui()
        self.setup_gstreamer()

    def setup_ui(self):
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        main_box.set_margin_top(10)
        main_box.set_margin_bottom(10)
        main_box.set_margin_start(10)
        main_box.set_margin_end(10)
        self.set_child(main_box)

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

        spectrum_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.spectrum_widget = SpectrumWidget()
        spectrum_box.append(self.spectrum_widget)
        notebook.append_page(spectrum_box, Gtk.Label(label="Spektrum i EQ"))

        bands_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.band_widgets = {}
        for band_name in self.analyzer.freq_bands.keys():
            frame = Gtk.Frame()
            frame.set_label(f"Pasmo: {band_name}")
            band_widget = Gtk.DrawingArea()
            band_widget.set_size_request(200, 100)
            band_widget.set_draw_func(self.draw_band_analysis, band_name)
            frame.set_child(band_widget)
            bands_box.append(frame)
            self.band_widgets[band_name] = band_widget
        scroll = Gtk.ScrolledWindow()
        scroll.set_child(bands_box)
        notebook.append_page(scroll, Gtk.Label(label="Analiza pasm"))

        self.band_weight_widget = BandWeightWidget(self.analyzer)
        scroll_weights = Gtk.ScrolledWindow()
        scroll_weights.set_child(self.band_weight_widget)
        notebook.append_page(scroll_weights, Gtk.Label(label="Wagi pasm"))

        self.report_view = Gtk.TextView()
        self.report_view.set_editable(False)
        self.report_view.set_wrap_mode(Gtk.WrapMode.WORD)
        report_scroll = Gtk.ScrolledWindow()
        report_scroll.set_child(self.report_view)
        notebook.append_page(report_scroll, Gtk.Label(label="Raport"))

        eq_frame = Gtk.Frame()
        eq_frame.set_label("Kontrola EQ (10 pasm)")
        main_box.append(eq_frame)
        eq_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        eq_frame.set_child(eq_box)
        self.eq_sliders = []
        eq_freqs = [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        for freq in eq_freqs:
            vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            slider = Gtk.Scale(orientation=Gtk.Orientation.VERTICAL)
            slider.set_range(-12, 12)
            slider.set_value(0)
            slider.set_inverted(True)
            slider.set_size_request(40, 150)
            slider.set_draw_value(True)
            slider.connect("value-changed", self.on_eq_changed)
            label = Gtk.Label(label=f"{freq}Hz")
            label.set_size_request(40, -1)
            vbox.append(slider)
            vbox.append(label)
            eq_box.append(vbox)
            self.eq_sliders.append(slider)
        reset_button = Gtk.Button(label="Reset EQ")
        reset_button.connect("clicked", self.on_reset_eq)
        eq_box.append(reset_button)

    def draw_band_analysis(self, area, cr, width, height, band_name):
        cr.set_source_rgb(0.1, 0.1, 0.1)
        cr.rectangle(0, 0, width, height)
        cr.fill()
        if hasattr(self, 'band_analysis') and band_name in self.band_analysis:
            band_data = self.band_analysis[band_name]
            cr.set_source_rgb(0.2, 0.8, 0.3)
            cr.set_line_width(2)
            mean_db = 20 * np.log10(band_data['mean'] + 1e-10)
            y_mean = height * (1 - (mean_db + 40) / 50)
            cr.move_to(0, y_mean)
            cr.line_to(width, y_mean)
            cr.stroke()
            peak_db = 20 * np.log10(band_data['peak'] + 1e-10)
            y_peak = height * (1 - (peak_db + 40) / 50)
            cr.set_source_rgb(1.0, 0.2, 0.2)
            cr.move_to(0, y_peak)
            cr.line_to(width, y_peak)
            cr.stroke()

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
            if struct.get_name() == "spectrum":
                magnitudes = struct.get_value("magnitude")
                spectrum = np.array(magnitudes, dtype=np.float32)
                freqs = np.linspace(0, self.analyzer.sample_rate / 2, len(spectrum))
                band_analysis, _, _ = self.analyzer.analyze_spectrum(spectrum)
                for band_name, band_widget in self.band_widgets.items():
                    GLib.idle_add(band_widget.queue_draw)
                GLib.idle_add(self.spectrum_widget.update_spectrum, spectrum, freqs)
                self.band_analysis = band_analysis
                imperfections = self.analyzer.detect_imperfections(band_analysis)
                eq_freqs, eq_gains = self.analyzer.generate_eq_curve(imperfections)
                GLib.idle_add(self.spectrum_widget.update_eq_curve, eq_freqs, eq_gains)
                GLib.idle_add(self.spectrum_widget.update_imperfections, imperfections)
                for i, gain in enumerate(eq_gains):
                    GLib.idle_add(self.eq_sliders[i].set_value, gain)
                    self.equalizer.set_property(f"band{i}", gain)

    def on_play_pause(self, widget):
        """Rozpoczyna/zatrzymuje nasłuchiwanie"""
        if not self.is_playing:
            self.pipeline.set_state(Gst.State.PLAYING)
            self.play_button.set_label("Zatrzymaj")
            self.is_playing = True
        else:
            self.pipeline.set_state(Gst.State.PAUSED)
            self.play_button.set_label("Nasłuchuj")
            self.is_playing = False

    def on_analyze(self, widget):
        """Przeprowadza analizę bieżącego spektrum"""
        if not hasattr(self, 'band_analysis'):
            self.status_label.set_text("Brak danych do analizy")
            return
        self.status_label.set_text("Analizowanie...")
        imperfections = self.analyzer.detect_imperfections(self.band_analysis)
        eq_freqs, eq_gains = self.analyzer.generate_eq_curve(imperfections)
        report = "=== RAPORT ANALIZY AUDIO ===\n\n"
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
        if hasattr(self, 'band_analysis') and hasattr(self, 'band_weight_widget'):
            imperfections = self.analyzer.detect_imperfections(self.band_analysis)
            weights = self.band_weight_widget.get_weights()
            eq_freqs, eq_gains = self.analyzer.generate_eq_curve(imperfections, weights)
            for i, (slider, gain) in enumerate(zip(self.eq_sliders, eq_gains)):
                slider.set_value(gain)
                self.equalizer.set_property(f"band{i}", gain)
            self.status_label.set_text("Zastosowano Auto-EQ z wagami pasm")

    def on_eq_changed(self, slider):
        """Callback dla zmian sliderów EQ"""
        index = self.eq_sliders.index(slider)
        value = slider.get_value()
        self.equalizer.set_property(f"band{index}", value)
        if hasattr(self, 'last_eq_curve'):
            eq_freqs, eq_gains = self.last_eq_curve
            eq_gains[index] = value
            self.spectrum_widget.update_eq_curve(eq_freqs, eq_gains)

    def on_reset_eq(self, widget):
        """Resetuje ustawienia EQ"""
        for i, slider in enumerate(self.eq_sliders):
            slider.set_value(0)
            self.equalizer.set_property(f"band{i}", 0)
        self.status_label.set_text("EQ zresetowany")

def main():
    # Utwórz wirtualne wyjście audio
    create_virtual_sink()

    # Uruchom aplikację GTK
    app = Gtk.Application(application_id="org.example.audioanalyzer")
    app.connect("activate", lambda a: AudioAnalyzerApp(application=a).present())
    app.run(None)

if __name__ == "__main__":
    main()
