# Phase 3 Empirical Grounding

This document anchors each Phase 3 phenomenon (pathology and healing dynamics) to
a *real, measured* dynamical signature from the peer-reviewed literature. The goal
is **structural grounding**: ensuring the CLT simulators reproduce the right kind
of observable and that regime assignments baked into the code are defensible.

This is deliberately **not** quantitative validation (fitting specific datasets),
which is deferred to Phase 4 / LoomSense.

## The synthesis

The CLT "viable window between chaos and rigidity" is, almost one-to-one, the
mainstream **brain-criticality / edge-of-chaos** hypothesis: healthy cognition sits
near a critical point between rigid order and noise, where complexity and
flexibility peak. Two observables recur across nearly every Phase 3 phenomenon and
are implemented as first-class metrics in `analysis/metrics`:

- **Signal complexity / diversity** (Lempel–Ziv, spectral entropy). High in
  conscious/healthy/wake/REM/psychedelic states; low in anesthesia, deep sleep, and
  depression. Maps onto the regime axis: *low complexity → rigidity*, *excess
  entropy → chaos*, *peak flexible complexity → viable window (near-criticality)*.
- **Critical slowing down** (rising lag-1 autocorrelation + variance): the generic
  early-warning signal as a system nears a tipping point.

## Grounding table

| Phase | Phenomenon | Real measured signature | Standard metric(s) | CLT regime | Sim target |
|---|---|---|---|---|---|
| 3.1 | Viable window | Brain near a critical point ("edge of chaos"); power-law neuronal avalanches | Avalanche size distributions, branching ratio ≈ 1 | Viable = near-criticality; chaos = supercritical, rigidity = subcritical | Attractor near a critical point between two failure modes |
| 3.1 | Regime transition | Critical slowing down before transitions | Lag-1 autocorrelation ↑, variance ↑ | Approach to either boundary | Rising AC(1)+variance as a control parameter drives toward a boundary |
| 3.1 | Hysteresis | Anesthesia **neural inertia**: dose to lose consciousness > dose to regain; asymmetric loss/recovery | Hysteresis loop in state vs. drive | Path-dependence of boundary crossing | Reproduce a hysteresis loop (induction threshold > release threshold) |
| 3.2 | Seizure | **Not** simple hypersync — onset often *desynchronization* → chimera-like partial sync → late hypersync | Kuramoto R, chimera/amplitude entropy | Dynamic breakdown of near-critical balance (not a static rigidity point) | Kuramoto network swinging through desync and hypersync phases |
| 3.2 | Depression | Reduced complexity; bistable mood with tipping points; CSD precedes onset/termination | LZ/entropy ↓; AC(1)+variance ↑ | Rigidity (fragmentation, reduced flux) | Bistable mood attractor + CSD early warning + complexity drop |
| 3.2 | Anesthesia | Complexity collapses; AC ↑ & connectivity ↓ toward unconsciousness; PCI drops | PCI, LZc, ΦG | Global decoupling → rigidity | Complexity/PCI-proxy collapse as drive rises; hysteretic recovery |
| 3.2 | Neurodegeneration | Progressive loss of EEG/MEG coherence & connectivity (esp. alpha/beta) | Coherence ↓, graph metrics, long-range FC ↓ | Slow contraction of coherence domain / volume V | Gradual coupling decay → shrinking Q / C_bio |
| 3.3 | Meditation | Long-term meditators self-induce high-amplitude **gamma synchrony** + frontoparietal coherence | Gamma power, phase-synchrony | Coherence enhancement within the viable window | Controlled coherence increase that stays inside the window |
| 3.3 | Psychedelics | **Increased** signal diversity/entropy ("entropic brain"); correlates with subjective intensity | LZc/LZs ↑, ACE/SCE ↑ | Boundary dissolution → chaos side | Raised entropy/diversity + softened boundaries |
| 3.3 | Sleep/wake | Complexity graded: wake ≈ REM (high) > N1 > deep sleep (low); reversible | LZ complexity, spectral slope | Cyclic traversal of the regime axis | Slow cyclic trajectory; complexity tracks the cycle |
| 3.3 | Therapeutic intervention | Re-coupling; healing settles into a *modified*, more resilient attractor | Recovery of coherence/complexity; changed basin | Restored coupling, deepened basins, expanded domain (§7.7) | Injury/decohere → intervention → recovery to a re-coupled attractor |

## Key tensions the literature resolved

1. **Seizure ≠ pure rigidity/hypersync.** The classic hypersynchrony view is
   incomplete: seizure *onset* is often desynchronization, passing through a
   chimera-like partial-sync stage. Model seizure as a *dynamic loss of the flexible
   near-critical regime*, not a static high-éR point.
2. **Critical slowing down is contested for seizures** (both positive results and a
   prominent "no evidence" paper), but well-supported for anesthesia induction and
   depression. The CSD detector is therefore a *hypothesis-testing* tool: demonstrate
   it on the clean cases and present seizure CSD as open.
3. **Hysteresis has a quantitative anchor** in anesthesia neural inertia (induction
   threshold > emergence threshold) — the `RegimeSystem` reproduces a *measured*
   loop shape.
4. **Complexity (LZ/entropy) is the unifying observable** and is now a shared metric.

## Representative sources

- Criticality / edge of chaos: Hesse & Gross 2014, *Front. Syst. Neurosci.*; "The
  human brain is on the edge of chaos" (Cambridge).
- Critical slowing down, seizures (both sides): Maturana et al. 2020, *Nat. Commun.*
  (biomarker for seizure susceptibility); Wilkat et al. 2019, *Chaos* (no evidence).
- Critical slowing down, depression: van de Leemput et al. 2014, *PNAS*.
- Seizure sync/desync: Jiruska et al. 2013, *J. Physiol.*; a modified-Kuramoto
  seizure model, 2023.
- Anesthesia complexity: Schartner et al. 2015 (propofol, *PLOS One*);
  Perturbational Complexity Index (Casali et al. 2013).
- Anesthesia hysteresis / neural inertia: Friedman et al. 2010, *PLOS One*;
  asymmetric loss/recovery of consciousness, 2021.
- Neurodegeneration: EEG-coherence markers in Alzheimer's disease (reviews, 2022).
- Meditation gamma synchrony: Lutz et al. 2004, *PNAS*.
- Psychedelic signal diversity: Schartner et al. 2017, *Sci. Rep.*
- Sleep complexity: spectral slope + Lempel–Ziv across sleep stages (*eNeuro* 2024).
