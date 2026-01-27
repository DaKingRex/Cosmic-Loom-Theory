# LoomSense v2 — Technical Spec Sheet

## PIC-Enhanced Prototype: GHz + Multi-Channel Optical Layer

---

## 1. Purpose

LoomSense v2 expands the v1 architecture into Gigahertz-band sensing and photonic integration, enabling detection of deeper biological coherence layers related to:

- GHz microtubule water-core waveguide modes
- cross-microtubule synchrony
- nonlinear mixing signatures
- multi-channel optical separation

This version introduces PIC (Photonic Integrated Circuit) elements to begin the transition toward the full LoomSense v3 platform.

---

## 2. Target Biological Signals

**Primary:**
- 0.5–20 GHz dielectric + microwave signatures
- GHz–MHz nonlinear mixing (downconverted signals readable at MHz)
- intra-microtubule synchrony patterns

**Secondary:**
- multi-channel biophoton intensity
- coarse spectral separation via PIC waveguides

---

## 3. Frequency Bands Covered

| Band | Target |
|------|--------|
| kHz–MHz | full inherited coverage from v1 |
| GHz (primary) | microtubule hollow-core modes, water-structured waveguide oscillations, long-range coherence signatures within dendritic networks |
| Biophotons | multi-waveguide channel routing (still intensity-based, not yet sub-nm spectroscopy) |

---

## 4. Detection Modalities

### A. GHz Dielectric / Microwave Sensing (Primary)

- microwave dielectric probes
- GHz resonant cavity or planar waveguide couplers
- sub-thermal GHz scattering measurements
- heterodyne downconversion to MHz for FPGA processing

### B. PIC-Based Multi-Channel Optical Routing

- integrated photonic waveguides (SiN / SiO₂)
- coarse wavelength-division multiplexing (CWDM)
- up to 4–16 optical channels (pre-spectral version)
- avalanche photodiodes or mini-SPAD arrays

### C. Nonlinear Mixing

- detection of GHz×MHz beat frequencies
- reveals hidden GHz resonances through MHz analysis
- massively boosts detectability without bulky microwave hardware

---

## 5. Hardware Architecture

- GHz dielectric/microwave probe + coupler
- PIC platform (SiN or hybrid SiN/Si)
- multi-channel optical routing through waveguides
- GHz LNA (low-noise amplifier) front-end
- heterodyne mixer + MHz-band digitizer
- 2–16 channel optical APD array
- FPGA/SoC for parallel processing

---

## 6. Expected Performance Targets

### GHz Layer

| Parameter | Target |
|-----------|--------|
| Sensitivity | detect GHz dielectric variations at milliwatt → microwatt scale |
| Bandwidth | 0.5–20 GHz |
| Frequency resolution (after mixing) | 1–50 kHz |
| Nonlinear mixing products | readable through v1's MHz pipeline |

### Optical Layer

| Parameter | Target |
|-----------|--------|
| Channel count | 4–16 waveguide channels |
| Photon sensitivity | above dark noise |
| Temporal resolution | ~500 ns → 10 µs |
| Channel isolation | clean separation |

---

## 7. Data Output Format

- GHz resonance maps (via downconversion)
- cross-frequency coupling matrices (GHz↔MHz)
- multi-channel optical intensity traces
- coherence stability metrics
- raw microwave scattering curves

---

## 8. Key Research Questions v2 Will Answer

- Can we detect GHz microtubule hollow-core modes non-invasively?
- Are GHz coherence patterns altered by anesthesia, psychedelics, meditation, fatigue, injury?
- Can nonlinear mixing reveal GHz resonances using MHz hardware?
- Do multi-channel optical fluctuations correlate with GHz peaks or state transitions?
- Can optical + GHz data be merged into a single multi-dimensional feature space?

---

## 9. Dependencies

- Photonic Integrated Circuit design & fabrication
- GHz couplers, resonators, and mixers
- multi-channel optical calibration
- materials selection (SiN preferred for biophoton routing)
- thermal stabilization

---

## 10. Forward Compatibility

v2 sets the foundation for LoomSense v3, enabling:

- full spectral range (400–1300 nm)
- sub-nm resolution
- picosecond correlations
- 100+ channel architectures
- SPAD-level single-photon detection
