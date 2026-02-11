import React, { useState, useEffect, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Stars, Html } from '@react-three/drei';
import * as THREE from 'three';

// Color scale for MSD Exponent (Diffusion Regime)
function getDiffusionColor(msd) {
  if (msd < 0.5) return new THREE.Color('#E74C3C'); // Sub-diffusive (Red)
  if (msd < 1.0) return new THREE.Color('#F39C12'); // Brownian (Orange)
  if (msd < 2.0) return new THREE.Color('#2ECC71'); // Super-diffusive (Green)
  return new THREE.Color('#9B59B6'); // Hyper-diffusive (Purple)
}

function Trajectory({ layers, meta, showMetrics }) {
  const curve = useMemo(() => {
    if (!layers || layers.length === 0) return null;

    const points = layers.map((layer, i) => {
      const pca = layer.pca_path;
      const cx = pca.reduce((s, p) => s + p[0], 0) / pca.length;
      const cy = i * 2;
      const cz = pca.reduce((s, p) => s + p[2], 0) / pca.length;
      return new THREE.Vector3(cx * 10, cy, cz * 10);
    });

    return new THREE.CatmullRomCurve3(points);
  }, [layers]);

  return (
    <group>
      {curve && (
        <mesh>
          <tubeGeometry args={[curve, 64, 0.15, 8, false]} />
          <meshStandardMaterial color="#666" transparent opacity={0.4} />
        </mesh>
      )}

      {layers.map((layer, i) => {
        const pca = layer.pca_path;
        const cx = pca.reduce((s, p) => s + p[0], 0) / pca.length * 10;
        const cy = i * 2;
        const cz = pca.reduce((s, p) => s + p[2], 0) / pca.length * 10;

        const rg = layer.metrics?.radius_of_gyration || 0.1;
        const msd = layer.metrics?.msd_exponent || 1.0;
        const eff_dim = layer.metrics?.effective_dim || 1.0;

        const radius = showMetrics.rg ? Math.max(0.2, rg * 0.8) : 0.3;
        const color = showMetrics.msd ? getDiffusionColor(msd) : new THREE.Color('#888');
        const opacity = showMetrics.effDim ? 0.5 + (eff_dim / 25) : 0.7;

        return (
          <mesh key={i} position={[cx, cy, cz]}>
            <sphereGeometry args={[radius, 12, 12]} />
            <meshStandardMaterial
              color={color}
              transparent
              opacity={Math.min(opacity, 0.9)}
              roughness={0.3}
              metalness={0.1}
            />
          </mesh>
        );
      })}
    </group>
  );
}

function Scene({ trajectory, showMetrics, cameraSync }) {
  if (!trajectory) {
    return (
      <mesh position={[0, 25, 0]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color="gray" wireframe />
      </mesh>
    );
  }

  return (
    <>
      <PerspectiveCamera makeDefault position={[40, 30, 40]} fov={50} />
      <OrbitControls target={[0, 25, 0]} enableDamping dampingFactor={0.05} />

      <ambientLight intensity={0.4} />
      <directionalLight position={[10, 50, 20]} intensity={0.8} />
      <Stars radius={200} depth={50} count={3000} factor={3} saturation={0} fade />

      <gridHelper args={[80, 80, 0x333333, 0x111111]} position={[0, -2, 0]} />

      <Trajectory layers={trajectory.layers} meta={trajectory} showMetrics={showMetrics} />

      {/* Group Label */}
      <Html position={[0, 52, 0]} center>
        <div style={{
          color: 'white',
          background: 'rgba(0,0,0,0.7)',
          padding: '8px 16px',
          borderRadius: '4px',
          fontSize: '14px',
          fontWeight: 'bold'
        }}>
          {trajectory.group} • {trajectory.condition} • {trajectory.correct ? '✓' : '✗'}
        </div>
      </Html>
    </>
  );
}

export default function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Comparison Mode
  const [comparisonMode, setComparisonMode] = useState(true);
  const [leftTraj, setLeftTraj] = useState(null);
  const [rightTraj, setRightTraj] = useState(null);

  // Metric Toggles
  const [showMetrics, setShowMetrics] = useState({
    rg: true,
    msd: true,
    effDim: false
  });

  useEffect(() => {
    console.log("Fetching trajectory data...");
    fetch('/trajectory_data.json')
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then(jsonData => {
        console.log("Data loaded:", jsonData.trajectories?.length || 0, "trajectories");
        if (!jsonData.trajectories) throw new Error("Invalid JSON");
        setData(jsonData);

        // Auto-select first G4 and first G1
        const g4 = jsonData.trajectories.find(t => t.group === 'G4');
        const g1 = jsonData.trajectories.find(t => t.group === 'G1');
        setLeftTraj(g4 || null);
        setRightTraj(g1 || null);

        setLoading(false);
      })
      .catch(err => {
        console.error("Load error:", err);
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const trajOptions = useMemo(() => {
    if (!data) return [];
    return data.trajectories.map((t, i) => ({
      value: i,
      label: `${t.id} (${t.group}, ${t.correct ? 'Correct' : 'Wrong'})`
    }));
  }, [data]);

  if (error) {
    return (
      <div style={{ color: 'red', padding: '40px', background: '#200', height: '100vh', fontFamily: 'monospace' }}>
        <h1>ERROR</h1>
        <p>{error}</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div style={{ color: 'white', padding: '40px', background: '#111', height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column' }}>
        <h1>Loading...</h1>
        <p>Fetching trajectory_data.json</p>
      </div>
    );
  }

  return (
    <div style={{ width: '100vw', height: '100vh', background: '#050505', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <div style={{
        background: '#111',
        borderBottom: '1px solid #333',
        padding: '12px 20px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        <div>
          <h2 style={{ margin: 0, color: '#eee', fontSize: '18px' }}>Trajectory Visualizer</h2>
          <p style={{ margin: '4px 0 0 0', color: '#888', fontSize: '12px' }}>
            Experiment 14 • {data?.trajectories?.length || 0} trajectories
          </p>
        </div>

        <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
          <label style={{ color: '#aaa', fontSize: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <input
              type="checkbox"
              checked={comparisonMode}
              onChange={e => setComparisonMode(e.target.checked)}
            />
            Split Screen
          </label>
        </div>
      </div>

      {/* Main Content */}
      <div style={{ flex: 1, display: 'flex' }}>
        {/* 3D Viewports */}
        <div style={{ flex: 1, display: 'flex' }}>
          {/* Left Viewport */}
          <div style={{ flex: 1, position: 'relative', borderRight: comparisonMode ? '1px solid #333' : 'none' }}>
            <Canvas>
              <Scene trajectory={leftTraj} showMetrics={showMetrics} cameraSync={false} />
            </Canvas>
            {leftTraj && (
              <div style={{ position: 'absolute', top: 10, left: 10, background: 'rgba(0,0,0,0.7)', padding: '8px', borderRadius: '4px', color: '#eee', fontSize: '11px' }}>
                <div><strong>Left:</strong> {leftTraj.id}</div>
                <div>Difficulty: {leftTraj.difficulty}</div>
              </div>
            )}
          </div>

          {/* Right Viewport (if comparison mode) */}
          {comparisonMode && (
            <div style={{ flex: 1, position: 'relative' }}>
              <Canvas>
                <Scene trajectory={rightTraj} showMetrics={showMetrics} cameraSync={false} />
              </Canvas>
              {rightTraj && (
                <div style={{ position: 'absolute', top: 10, left: 10, background: 'rgba(0,0,0,0.7)', padding: '8px', borderRadius: '4px', color: '#eee', fontSize: '11px' }}>
                  <div><strong>Right:</strong> {rightTraj.id}</div>
                  <div>Difficulty: {rightTraj.difficulty}</div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Control Panel */}
        <div style={{
          width: '320px',
          background: '#111',
          borderLeft: '1px solid #333',
          padding: '20px',
          overflowY: 'auto',
          color: '#eee',
          fontFamily: 'sans-serif'
        }}>
          <h3 style={{ marginTop: 0 }}>Trajectory Selection</h3>

          <div style={{ marginBottom: '20px' }}>
            <label style={{ display: 'block', marginBottom: '8px', fontSize: '13px', color: '#aaa' }}>
              Left Viewport:
            </label>
            <select
              value={data.trajectories.indexOf(leftTraj)}
              onChange={e => setLeftTraj(data.trajectories[e.target.value])}
              style={{ width: '100%', padding: '8px', background: '#222', color: '#eee', border: '1px solid #444', borderRadius: '4px' }}
            >
              {trajOptions.map(opt => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>

          {comparisonMode && (
            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', marginBottom: '8px', fontSize: '13px', color: '#aaa' }}>
                Right Viewport:
              </label>
              <select
                value={data.trajectories.indexOf(rightTraj)}
                onChange={e => setRightTraj(data.trajectories[e.target.value])}
                style={{ width: '100%', padding: '8px', background: '#222', color: '#eee', border: '1px solid #444', borderRadius: '4px' }}
              >
                {trajOptions.map(opt => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
            </div>
          )}

          <hr style={{ border: 'none', borderTop: '1px solid #333', margin: '20px 0' }} />

          <h3>Metric Visualizations</h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px' }}>
              <input
                type="checkbox"
                checked={showMetrics.rg}
                onChange={e => setShowMetrics({ ...showMetrics, rg: e.target.checked })}
              />
              <span>Radius of Gyration (Size)</span>
            </label>

            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px' }}>
              <input
                type="checkbox"
                checked={showMetrics.msd}
                onChange={e => setShowMetrics({ ...showMetrics, msd: e.target.checked })}
              />
              <span>MSD Exponent (Color)</span>
            </label>

            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px' }}>
              <input
                type="checkbox"
                checked={showMetrics.effDim}
                onChange={e => setShowMetrics({ ...showMetrics, effDim: e.target.checked })}
              />
              <span>Effective Dim (Opacity)</span>
            </label>
          </div>

          <hr style={{ border: 'none', borderTop: '1px solid #333', margin: '20px 0' }} />

          <h3>Legend</h3>
          <div style={{ fontSize: '12px', lineHeight: '1.8', color: '#aaa' }}>
            <div><strong style={{ color: '#eee' }}>Vertical Axis:</strong> Layer (0-24)</div>
            <div><strong style={{ color: '#eee' }}>Horizontal:</strong> PCA State Space</div>
            <div><strong style={{ color: '#eee' }}>Tube:</strong> Trajectory Backbone</div>
            <div style={{ marginTop: '12px' }}>
              <div style={{ color: '#E74C3C' }}>■ Red: Ballistic (&lt;0.5)</div>
              <div style={{ color: '#F39C12' }}>■ Orange: Brownian (0.5-1.0)</div>
              <div style={{ color: '#2ECC71' }}>■ Green: Super-diffusive (1-2)</div>
              <div style={{ color: '#9B59B6' }}>■ Purple: Hyper-diffusive (&gt;2)</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
