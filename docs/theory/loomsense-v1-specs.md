# LoomSense v1 — Technical Spec Sheet

## Feasibility Prototype: MHz + Biophoton Layer

---

## 1. Purpose

LoomSense v1 is the first-stage feasibility prototype designed to detect:

- MHz-range biological resonance signatures
- ultraweak biophoton emissions (intensity + temporal patterns)

This phase establishes early biological correlations and provides foundational data for engineering iterations and scientific validation.

---

## 2. Target Biological Signals

**Primary:**
- 1–100 MHz microtubule-associated oscillations
- MHz triplet structures discovered in microtubule signatures

**Secondary:**
- interference patterns linked to EEG state transitions
- ultraweak biophoton emissions (visible–NIR intensity + temporal fluctuation)

---

## 3. Frequency Bands Covered

| Band | Target |
|------|--------|
| kHz (indirect) | C-terminus / hydration layer |
| MHz (primary) | microtubule lattice phonons & polaritons |
| Biophotons | broadband 400–900 nm (intensity only in v1) |

---

## 4. Detection Modalities

### A. RF Dielectric Resonance (Primary)

- MHz dielectric coil sensors
- high-bandwidth (>20 MHz) preamplification
- low-noise analog front-end
- FPGA/SoC for real-time FFT + peak-extraction

### B. Ultraweak Optical Detection (Secondary)

- silicon photodiodes or avalanche photodiodes
- long-integration low-light measurement
- temporal fluctuation analysis (not full spectroscopy yet)

---

## 5. Hardware Architecture

- RF resonant coil array (1–100 MHz)
- transimpedance amplifiers (TIAs)
- adjustable gain stages (low-noise)
- 16–24 bit ADC
- FPGA or high-speed microcontroller
- optical low-noise detector (single-channel)
- thermally shielded dark enclosure (passive)

---

## 6. Expected Performance Targets

| Parameter | Target |
|-----------|--------|
| RF sensitivity | detect MHz peaks at microvolt–millivolt scale |
| Frequency resolution | ≤100 kHz |
| Photon sensitivity | detect ultraweak emission above ambient dark noise |
| Temporal resolution (optical) | 1–10 µs |

**Output:**
- triplet-band signatures
- FFT spectra
- temporal photon fluctuation curves

---

## 7. Data Output Format

- Raw MHz FFT data (CSV/JSON)
- Extracted triplet frequencies and harmonics
- Photon-intensity time series
- Session metadata (duration, gain, conditions)

---

## 8. Key Feasibility Questions v1 Will Answer

- Can we detect MHz triplet signatures non-invasively in tissue?
- How stable are these signatures across subjects?
- Do MHz patterns correlate with known physiological shifts (relaxation, focus, sleepiness)?
- Can ultraweak photon fluctuations track biological state changes?

---

## 9. Dependencies

- Coil design + RF tuning
- Low-noise electronics
- Light-tight enclosure
- Calibration routines

---

## 10. Forward Compatibility

v1 is explicitly designed so its RF + optical subsystems integrate directly into:

- v2 PIC microwave interface
- future multi-channel optical arrays
- software pipeline for higher-dimensional data
