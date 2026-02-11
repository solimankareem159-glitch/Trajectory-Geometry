# Visualization Tool Redesign Proposal

**Date:** 2026-02-10
**Design Objective:** Create an intuitive, shareable interface that makes the core findings (collapse, expansion, commitment) visually self-evident to a general ML audience.

---

## 1. Assessment of Current Design

The existing design document (`visualisation_tool/design.md`) outlines a comprehensive 3D exploration suite.

### Strengths
*   **Pipeline Architecture:** Solid data flow (JSONL → React State).
*   **Metric Encoding:** Correctly maps $R_g$ to sphere size and MSD to color.
*   **Multi-View:** Conceptually supports both "God View" (PCA) and "First Person" (Flythrough).

### Critical Gaps
*   **Cognitive Load:** 3D PCA clouds are famously difficult to interpret on 2D screens without constant rotation.
*   **Invisible Boundaries:** The current design plots trajectories but doesn't explicitly visualize the *Regime Boundaries* (the "walls" between success and failure).
*   **Missing Commitment Signal:** The "Time to Commit" metric is calculated but not effectively visualized as a discrete event in the current 3D tube rendering.

---

## 2. Proposed Redesign: The "Layered Insight" Architecture

We propose shifting from a "Sandbox" model to a "Guided Lens" model with three synchronized views.

### 2.1 View A: The Regime Map (2D Context)
**Concept:** A top-down 2D projection (PCA 1 vs PCA 2) of the *entire dataset* at a specific layer depth.
*   **Visual:** A density map (contour plot) showing the "Safe Zone" (Direct Success cluster) and the "Wilderness" (CoT Exploration zone).
*   **Interaction:** Brushing. Select a region to isolate those trajectories in the 3D view.
*   **Why:** Immediately shows "Dimensional Collapse" as a tight knot vs. "Wandering" as a diffuse cloud.

### 2.2 View B: The Trajectory Theater (3D Focus)
**Concept:** The main stage. Renders selected trajectories as 3D tubes.
*   **Improvement:** **Regime Walls**. Render semi-transparent isosurfaces that bounded the "Success Manifold." If a trajectory punches through the wall, it fails.
*   **Metaphor:**
    *   **Direct Success:** A straight, efficient arrow hitting the target.
    *   **CoT Success:** A wide spiraling arc that eventually snaps to the target.
    *   **Wandering Failure:** A spiral that expands but misses the target.
    *   **Collapsed Failure:** A short, crumpled line that falls short.

### 2.3 View C: The Commitment Scanner (1D Temporal)
**Concept:** A "Sparkline" view aligned with the token sequence.
*   **X-Axis:** Token Index (0 to 32).
*   **Y-Axis:** Radius of Gyration ($R_g$) or Entropy.
*   **Visual:**
    *   **Exploration Phase:** High, volatile line (Red/Orange).
    *   **Commitment Point:** Vertical marker where the line crashes down.
    *   **Execution Phase:** Low, flat line (Blue).
*   **Why:** Makes the "Phase Transition" finding explicit.

---

## 3. Interaction & Metaphors

To communicate these abstract concepts to a non-expert, we will use consistent metaphors:

| Concept | Visual Metaphor | Encoding |
| :--- | :--- | :--- |
| **Reasoning Effort** | **Heat / Inflation** | Sphere Size ($R_g$) + Color (Red=Hot/Thinking, Blue=Cool/Done) |
| **Commitment** | **Crystallization** | Sudden transition from chaotic curve to straight line |
| **Failure** | **Dispersion** | Fading opacity + "Lost" icon at end of path |
| **Success** | **Target Lock** | Glowing ring at the destination point |

**Linked Brushing:**
*   Hover over a token in the *Scanner* $\to$ Highlight that point on the *3D Trajectory*.
*   Click a cluster in the *Regime Map* $\to$ Load those paths into the *Theory*.

---

## 4. Minimum Viable Demo (For Medium Article)

We do not need the full tool for the article. We need a **"Scrollytelling"** component.

**Deliverable:** A single `index.html` build that:
1.  **Scene 1:** Shows 10 "Direct Success" paths (Straight, Blue). Text: *"Standard models answer directly."*
2.  **Scene 2:** Fades in 10 "CoT Success" paths (Wide Arcs, Orange). Text: *"Thinking models expand their state space."*
3.  **Scene 3:** Overlays "CoT Failures" (Wide Arcs, dimmer). Text: *"But sometimes they get lost."*
4.  **Scene 4:** Highlights the **Commitment Point**. Text: *"Ideally, they collapse to a solution here."*

**Tech Stack:**
*   Three.js (Canvas)
*   ScrollMagic (Triggering scenes based on scroll)
*   React-Three-Fiber (Declarative scene graph)

This is far more effective than a generic dashboard for a general audience.
