# Computational Trajectory Visualization Tool: Design Document

## Vision Statement

An **interactive 3D visualization** that transforms high-dimensional neural network trajectories into intuitive, explorable visual metaphors, enabling researchers to **see** how models think—distinguishing reasoning from retrieval, success from failure, and exploration from commitment.

**Data Foundation**: Grounded in **Experiment 14** (Qwen 0.5B, 600 samples, 31 metrics) and **Experiment 15 extensions** (difficulty analysis, failure clustering, token dynamics). This clean, complete dataset provides robust geometric signatures without hallucination artifacts.

---

## 1. Conceptual Framework

### The Core Metaphor: "Thought as Journey"

Each computation is a **journey through conceptual space**:
- **Vertical axis**: Layer depth (0 → 24, Qwen 0.5B's 25 layers including embeddings)
- **Horizontal plane**: Conceptual exploration (PCA projection of 896-dim hidden states)
- **Temporal progression**: Token sequence (32 tokens per trajectory)
- **Visual artifacts**: 31 geometric metrics encoded as shape, color, motion, and texture

### Key Interaction Paradigm

Users **fly through** the computational space, comparing trajectories side-by-side, toggling metrics on/off, and discovering patterns through embodied exploration.

---

## 2. Visual Encoding System

### 2.1 Trajectory Path (Core Element)

**Representation**: 3D tube with variable properties

| Metric | Visual Encoding |
|:---|:---|
| **Layer Position** | Vertical position (Y-axis: 0-24 layers) |
| **Hidden State (PCA)** | XZ position (2D projection of 896-dim space) |
| **Token Position** | Opacity/saturation (fades toward end of 32-token fixed window) |
| **Speed** | Spacing between nodes (wider = faster movement) |
| **Directional Consistency** | Line opacity (solid = consistent, faded = erratic) |

### 2.2 Expansion/Contraction Metaphor

**Representation**: Animated spheres/ellipsoids at each layer node

| Metric | Visual Encoding |
|:---|:---|
| **Radius of Gyration** | Sphere size (larger = more exploration) |
| **Effective Dimension** | Particle cloud density inside sphere |
| **Gyration Anisotropy** | Sphere → Ellipsoid deformation |
| **Drift to Spread** | Direction of ellipsoid elongation |

**Animation**: Spheres **pulse** with breathing rhythm, expanding during exploration, contracting during commitment.

### 2.3 Diffusion Regime Encoding

**Representation**: Color gradient applied to trajectory tube

| MSD Exponent Range | Color | Meaning |
|:---|:---|:---|
| **< 0.5** | Deep Red | Sub-diffusive (trapped/ballistic) |
| **0.5 - 1.0** | Orange → Yellow | Brownian (random walk) |
| **1.0 - 2.0** | Green → Cyan | Super-diffusive (exploratory) |
| **> 2.0** | Blue → Purple | Hyper-diffusive (chaotic) |

**Gradient**: Color smoothly transitions along trajectory, revealing **phase transitions** mid-computation.

### 2.4 Alignment & Commitment

**Representation**: Directional arrows and glowing markers

| Metric | Visual Encoding |
|:---|:---|
| **Cosine to Final State** | Arrow pointing toward final state (opacity = alignment strength) |
| **Time to Commit** | Glowing sphere marker at commitment point |
| **Stabilization Rate** | Trajectory tube taper (narrowing = stabilizing) |

### 2.5 Recurrence & Memory

**Representation**: Ghost echoes and connection threads

| Metric | Visual Encoding |
|:---|:---|
| **Recurrence Rate** | Translucent "ghost" copies of spheres at similar states |
| **Laminarity** | Vertical threads connecting recurrent states |
| **Trapping Time** | Pulsing rings around trapped regions |

### 2.6 Spectral & Frequency Features

**Representation**: Glowing aura and sound

| Metric | Visual Encoding |
|:---|:---|
| **Spectral Entropy** | Chromatic aberration/prismatic glow around trajectory |
| **PSD Slope** | Audio tone (procedurally generated from trajectory) |

---

## 3. Comparison & Contrast Modes

### 3.1 Side-by-Side Comparison

**Layout**: Split screen showing 2-4 trajectories simultaneously
- **Left**: CoT Success (G4) in **green-cyan palette**
- **Right**: Direct Fail (G1) in **red-orange palette**

**Camera**: Synchronized rotation/zoom, or independent controls

### 3.2 Overlay Mode

**Ghosts**: Overlay multiple trajectories in same space with transparency
- **Cluster view**: Show 10-20 trajectories from same group (e.g., all G4) to reveal **population patterns**
- **Outlier detection**: Highlight trajectories that deviate from cluster centroid

### 3.3 Difference Heatmap

**Projection**: Project trajectory onto 2D plane, color by metric difference
- **Example**: Show G4-G1 difference in effective dimension across layers

---

## 4. Interactive Features

### 4.1 Core Interactions

1. **Camera Control**
   - Orbit, pan, zoom with mouse/trackpad
   - Preset views: "Side view" (layer progression), "Top view" (exploration plane), "Commitment view" (focus on time-to-commit marker)

2. **Trajectory Selection**
   - Click to select sample from dataset
   - Filter by group (G1-G4), correctness, problem ID
   - Random sampling within group

3. **Metric Toggles**
   - Checkboxes to enable/disable visual encodings
   - Slider to control visual intensity (e.g., sphere size multiplier)

4. **Playback Animation**
   - "Play" button to animate trajectory growth token-by-token
   - Speed control (1x - 100x)
   - Pause at specific layers

### 4.2 Educational Modes

1. **Guided Tour**
   - Scripted camera path with narration
   - Highlights key differences between G4 and G1
   - Teaches metric interpretation

2. **Quiz Mode**
   - Show unlabeled trajectory, user guesses group (G1-G4)
   - Immediate feedback with metric breakdown
   - Leaderboard for pattern recognition skill

3. **Exploration Sandbox**
   - Load any sample, freely toggle all metrics
   - Export camera path as video
   - Annotation mode (add text labels to space)

### 4.3 Data Query Panel

**UI**: Side panel with controls
- **Group filter**: Checkboxes for G1, G2, G3, G4
- **Metric range sliders**: Filter trajectories by metric values
   - Example: "Show only samples with effective_dim > 10 at layer 12"
- **Problem selection**: Dropdown or search by problem ID
- **Comparison builder**: Drag-and-drop to create custom comparisons

---

## 5. Technical Architecture

### 5.1 Technology Stack

**Recommended**: Three.js (WebGL) + React for UI

| Component | Technology | Rationale |
|:---|:---|:---|
| **3D Rendering** | Three.js | Industry-standard WebGL library, excellent performance |
| **UI Framework** | React | Component-based, easy state management |
| **Data Pipeline** | Python (preprocessing) → JSON | Load metrics CSV, compute PCA, export to JSON |
| **Dimensionality Reduction** | PCA (2-3 components) | Project 896-dim hidden states to 3D space |
| **Animation** | GSAP (GreenSock) | Smooth transitions, timeline control |
| **Audio** | Web Audio API | Procedural sound generation from metrics |

### 5.2 Data Pipeline

```
Experiment 14/15 Data (CSV + NPY) [Qwen 0.5B]
    ↓
[Python Preprocessing Script]
    - Load exp14_metrics.csv (15,600 rows)
    - Load exp14_comparisons.csv (statistical results)
    - Load hidden_states/*.npy (600 files)
    - Load exp15 extensions (difficulty, clusters, windows)
    - Compute PCA on hidden states (896D → 3D)
    - Package metrics + PCA coords + extensions into JSON
    ↓
trajectory_data.json
    ↓
[Three.js Visualization]
    - Load JSON
    - Render trajectories with all 31 metrics
    - Apply visual encodings
```

### 5.3 JSON Data Structure

```json
{
  "trajectories": [
    {
      "id": "problem_000_cot",
      "group": "G4",
      "correct": true,
      "condition": "cot",
      "difficulty_tier": "medium",
      "failure_cluster": null,
      "layers": [
        {
          "layer_idx": 0,
          "pca_coords": [0.5, 0.2, -0.3],
          "metrics": {
            "radius_of_gyration": 0.41,
            "effective_dim": 5.2,
            "msd_exponent": 1.2,
            "speed": 0.05,
            "spectral_entropy": 0.75,
            ...
          },
          "token_dynamics": {
            "window_rg_growth_rate": 0.02,
            "phase_transition_detected": false
          }
        },
        ...
      ]
    },
    ...
  ]
}
```

### 5.4 Performance Optimization

- **LOD (Level of Detail)**: Reduce sphere tessellation when zoomed out
- **Instancing**: Use GPU instancing for repeated geometry (spheres)
- **Culling**: Don't render trajectories outside camera frustum
- **Lazy loading**: Load trajectory data on-demand when selected
- **Web Workers**: Offload PCA computation to background thread

---

## 6. Visual Design Details

### 6.1 Color Palettes

**Group Palettes** (when showing single trajectory):

| Group | Primary Color | Gradient |
|:---|:---|:---|
| **G1** (Direct Fail) | Red (#E74C3C) | Red → Dark Red |
| **G2** (Direct Success) | Blue (#3498DB) | Blue → Light Blue |
| **G3** (CoT Fail) | Orange (#E67E22) | Orange → Dark Orange |
| **G4** (CoT Success) | Green (#2ECC71) | Green → Cyan |

**Metric Overlay** (when metric color overrides group color):
- Use HSL color space for smooth MSD exponent gradient
- Opacity modulates based on metric confidence

### 6.2 Lighting & Atmosphere

- **Ambient light**: Soft white (#FFFFFF, intensity 0.3)
- **Directional light**: Warm top-down (#FFEAA7, intensity 0.7)
- **Point lights**: Placed at commitment markers (glow effect)
- **Fog**: Subtle depth fog (far = 150 units) for depth perception
- **Skybox**: Dark gradient (top: deep blue, bottom: black)

### 6.3 Materials

- **Trajectory tube**: Phong material with specular highlights
- **Exploration spheres**: Glass-like material (transparency, refraction)
- **Commitment markers**: Emissive material (glowing)
- **Floor grid**: Subtle wireframe (helps spatial orientation)

---

## 7. Implementation Roadmap

### Phase 1: Data Preparation (1 week)

**Script**: `generate_visualization_data.py`

- [ ] Load `exp14_metrics.csv` (15,600 rows × 38 columns)
- [ ] Load `metadata.csv` (problem IDs, groups, correctness)
- [ ] Load all `.npy` files from `hidden_states/` (600 files)
- [ ] Load Exp 15 extended data:
  - [ ] `analysis_A_difficulty_results.csv` (difficulty tiers)
  - [ ] `analysis_B_centroids.csv` (failure clusters)
  - [ ] `analysis_C_window_metrics.csv` (token dynamics)
- [ ] Compute PCA (sklearn) on concatenated hidden states (896D → 3D)
- [ ] Merge all data sources by problem_id + condition
- [ ] Export `trajectory_data.json` with PCA coords + all metrics + extensions
- [ ] Validate JSON schema
- [ ] Generate summary statistics (avg metrics per group, PCA variance explained)

### Phase 2: Core 3D Engine (2 weeks)

**Tech**: Three.js + Vite

- [ ] Setup Vite + Three.js project
- [ ] Create `TrajectoryRenderer` class
- [ ] Implement basic trajectory path (tube geometry)
- [ ] Add orbit controls (camera)
- [ ] Load JSON data
- [ ] Render single trajectory with layer markers

### Phase 3: Visual Encodings (2 weeks)

- [ ] Implement sphere scaling (radius of gyration)
- [ ] Add color gradient (MSD exponent)
- [ ] Implement ellipsoid deformation (anisotropy)
- [ ] Add ghost echoes (recurrence)
- [ ] Create commitment marker (glowing sphere)
- [ ] Implement trajectory tube opacity (directional consistency)

### Phase 4: Comparison & UI (2 weeks)

**Tech**: React + Tailwind CSS

- [ ] Build React UI shell
- [ ] Create trajectory selection panel
- [ ] Implement side-by-side comparison mode
- [ ] Add overlay mode (ghost trajectories)
- [ ] Create metric toggle checkboxes
- [ ] Build filter controls (group, metrics)

### Phase 5: Animation & Interactivity (1 week)

- [ ] Implement playback timeline
- [ ] Add token-by-token animation
- [ ] Create preset camera views
- [ ] Implement smooth transitions (GSAP)
- [ ] Add hover tooltips (show metric values)

### Phase 6: Educational Features (1 week)

- [ ] Build guided tour script
- [ ] Implement quiz mode
- [ ] Create annotation system
- [ ] Add exportvideo feature (screen recording)

### Phase 7: Polish & Optimization (1 week)

- [ ] Performance profiling
- [ ] Implement LOD system
- [ ] Add loading screens
- [ ] Create help/tutorial overlay
- [ ] Write documentation

**Total**: 10 weeks for full implementation

---

## 8. Example Use Cases

### Use Case 1: Discovering "The Commitment Point"

**Scenario**: Researcher wants to understand when CoT trajectories "commit" to a solution.

**Workflow**:
1. Load 20 random G4 (CoT Success) trajectories in overlay mode
2. Enable "Time to Commit" marker visualization
3. Observe: Most markers cluster around layer 18-22
4. **Insight**: CoT commits late, after extensive exploration

### Use Case 2: Failure Mode Diagnosis

**Scenario**: User wants to understand why some CoT attempts fail (G3).

**Workflow**:
1. Load G3 trajectory, then load similar G4 trajectory side-by-side
2. Toggle on: Radius of Gyration, Effective Dimension, MSD Exponent
3. Observe: G3 has smaller spheres (less exploration) and red color (sub-diffusive)
4. **Insight**: G3 failures "collapse" prematurely, never entering exploratory phase

### Use Case 3: Teaching Neural Network Behavior

**Scenario**: Educator wants to teach students about CoT vs Direct answering.

**Workflow**:
1. Load G3 trajectory, then load similar G4 trajectory side-by-side
2. Toggle on: Radius of Gyration, Effective Dimension, MSD Exponent
3. Observe: G3 has smaller spheres (less exploration) and red color (sub-diffusive)
4. **Insight**: G3 failures "collapse" prematurely, never entering exploratory phase

---

## 9. Advanced Features (Future Extensions)

### 9.1 Real-Time Inference Visualization

- Connect to live model API
- Type in custom problem
- Watch trajectory form in real-time as model generates tokens

### 9.2 Trajectory Interpolation

- Morph between two trajectories (e.g., G1 → G4)
- Visualize "what would need to change" for failure to become success

### 9.3 VR Mode

- WebXR support for fully immersive exploration
- Grab and manipulate trajectories with VR controllers

### 9.4 Collaborative Exploration

- Multi-user mode (WebRTC)
- Shared camera, synchronized annotations
- Voice chat for remote research discussions

---

## 10. Success Metrics

The tool **succeeds** if:

1. **Pattern Recognition**: Users can distinguish G1 from G4 with >80% accuracy after 5 minutes of exploration
2. **Insight Generation**: Researchers report discovering **new hypotheses** about model behavior
3. **Educational Impact**: Students demonstrate improved understanding of neural network internals
4. **Publication Value**: Tool is cited in papers as a **standard visualization method**

---

## 11. Conclusion

This visualization tool transforms abstract geometric metrics into **embodied, explorable experiences**. By grounding every visual choice in actual data and encoding multiple metrics simultaneously, it enables **pattern discovery** that would be impossible with static plots.

The design respects the complexity of high-dimensional trajectories while making them **accessible** through carefully chosen metaphors: expansion as exploration, color as motion regime, position as conceptual location.
