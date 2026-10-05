/**
 * ============================================================================
 * Conway's Game of Life — High Performance Simulation & Visualization Engine
 * ============================================================================
 * Features:
 *   - High-performance TypedArray double-buffered compute pipeline
 *   - 32-bit direct pixel memory blitting via ImageData & Uint32Array
 *   - Interactive Pan & Zoom viewport with coordinate inspection
 *   - Cell longevity / aging gradient coloring
 *   - Preset patterns (Gosper Gun, Pulsar, Acorn, etc.) & 30% random initializer
 *   - Live HPC telemetry (gen/s, FPS, population %, throughput in M cells/s)
 * ============================================================================
 */

(function () {
  'use strict';

  /* ── State & Variables ─────────────────────────────────────────────────── */
  let N = 1024; // Default grid dimension (N x N)
  let currentGrid = new Uint8Array(N * N);
  let nextGrid = new Uint8Array(N * N);
  let ageGrid = new Uint16Array(N * N);

  let isRunning = false;
  let targetGenPerSec = 30;
  let generation = 0;
  let aliveCount = 0;
  let toroidal = true;
  let cellAging = true;
  let activeTheme = 'theme-emerald';
  let activeTool = 'draw'; // 'draw' | 'erase' | 'pan'

  /* Viewport Pan & Zoom */
  let zoom = 1.0;
  let panX = 0;
  let panY = 0;
  let isDragging = false;
  let dragStartX = 0;
  let dragStartY = 0;
  let isDrawingOnMove = false;

  /* Offscreen Canvas for Direct Pixel Blitting */
  let offCanvas = document.createElement('canvas');
  let offCtx = offCanvas.getContext('2d', { willReadFrequently: true });
  let offImageData = null;
  let offPixelBuf32 = null;

  /* Performance & Timing Counters */
  let lastFrameTime = performance.now();
  let lastGenTime = performance.now();
  let frameCount = 0;
  let genCount = 0;
  let fps = 60;
  let actualGenRate = 0;
  let throughput = 0; // Million cells / sec
  let genAccumulator = 0;

  /* ── DOM Elements ──────────────────────────────────────────────────────── */
  const canvas = document.getElementById('life-canvas');
  const ctx = canvas.getContext('2d');
  const viewport = document.getElementById('viewport-container');

  const valGen = document.getElementById('val-generation');
  const valAlive = document.getElementById('val-alive');
  const valAlivePct = document.getElementById('val-alive-pct');
  const valSimSpeed = document.getElementById('val-sim-speed');
  const valThroughput = document.getElementById('val-throughput');
  const valGridSize = document.getElementById('val-grid-size');
  const valFps = document.getElementById('val-fps');

  const btnPlayPause = document.getElementById('btn-play-pause');
  const iconPlay = document.getElementById('icon-play');
  const iconPause = document.getElementById('icon-pause');
  const labelPlayPause = document.getElementById('label-play-pause');
  const btnStep = document.getElementById('btn-step');
  const btnRandomize = document.getElementById('btn-randomize');
  const btnClear = document.getElementById('btn-clear');
  const btnResetView = document.getElementById('btn-reset-view');
  const btnFullscreen = document.getElementById('btn-fullscreen');

  const sliderSpeed = document.getElementById('slider-speed');
  const labelSpeedVal = document.getElementById('label-speed-val');
  const selectGridSize = document.getElementById('select-grid-size');
  const selectPreset = document.getElementById('select-preset');
  const selectTheme = document.getElementById('select-theme');
  const checkAging = document.getElementById('check-cell-aging');
  const checkToroidal = document.getElementById('check-toroidal');

  const hudZoom = document.getElementById('hud-zoom');
  const hudCoords = document.getElementById('hud-coords');
  const hudHoverState = document.getElementById('hud-hover-state');

  const toolDraw = document.getElementById('tool-draw');
  const toolErase = document.getElementById('tool-erase');
  const toolPan = document.getElementById('tool-pan');

  const btnToggleBenchmark = document.getElementById('btn-toggle-benchmark');
  const modalBenchmark = document.getElementById('modal-benchmark');
  const btnCloseBenchmark = document.getElementById('btn-close-benchmark');

  /* ── Theme Color Palettes (32-bit Little-Endian ABGR) ──────────────────── */
  // Format: 0xAABBGGRR
  const THEME_PALETTES = {
    'theme-emerald': {
      bg: 0xFF120a07, // #070a12
      young: 0xFF99e634, // #34e699 (Bright mint)
      mature: 0xFF81b910, // #10b981 (Emerald)
      old: 0xFF4ade80, // #80de4a (Lime)
      ancient: 0xFFbbf7d0, // #d0f7bb (Pale glow)
    },
    'theme-cyber': {
      bg: 0xFF140809,
      young: 0xFFf0d022, // Cyan
      mature: 0xFFd946ef, // Neon Magenta
      old: 0xFF8b5cf6, // Violet
      ancient: 0xFFffffff, // White Hot
    },
    'theme-thermal': {
      bg: 0xFF120703,
      young: 0xFFe6b800, // Deep Blue/Cyan
      mature: 0xFF38d010, // Green
      old: 0xFF00a5ff, // Orange
      ancient: 0xFF2222ef, // Red
    },
    'theme-amber': {
      bg: 0xFF0b080f,
      young: 0xFF16a34a,
      mature: 0xFF1673f9,
      old: 0xFF2626dc,
      ancient: 0xFFfef08a,
    },
    'theme-matrix': {
      bg: 0xFF061006,
      young: 0xFF5eead4,
      mature: 0xFF22c55e,
      old: 0xFF15803d,
      ancient: 0xFF86efac,
    }
  };

  /* ── Initialization ────────────────────────────────────────────────────── */
  function initGrid(size) {
    N = size;
    currentGrid = new Uint8Array(N * N);
    nextGrid = new Uint8Array(N * N);
    ageGrid = new Uint16Array(N * N);

    offCanvas.width = N;
    offCanvas.height = N;
    offImageData = offCtx.createImageData(N, N);
    offPixelBuf32 = new Uint32Array(offImageData.data.buffer);

    generation = 0;
    aliveCount = 0;
    valGridSize.textContent = `${N} × ${N}`;

    resetView();
    randomizeGrid(30);
  }

  function resetView() {
    const vWidth = viewport.clientWidth;
    const vHeight = viewport.clientHeight;
    // Fit grid nicely inside viewport with margins
    const margin = 40;
    const availWidth = vWidth - margin * 2;
    const availHeight = vHeight - margin * 2;
    zoom = Math.min(availWidth / N, availHeight / N);
    if (zoom <= 0) zoom = 1.0;

    panX = (vWidth - N * zoom) / 2;
    panY = (vHeight - N * zoom) / 2;
    updateZoomHUD();
  }

  function updateZoomHUD() {
    hudZoom.textContent = `Zoom: ${Math.round(zoom * 100)}%`;
  }

  /* ── Random Initializer (Matches C assignment rand() % 100 < 30) ───────── */
  function randomizeGrid(densityPercent = 30) {
    aliveCount = 0;
    for (let i = 0; i < N * N; i++) {
      const isAlive = Math.random() * 100 < densityPercent ? 1 : 0;
      currentGrid[i] = isAlive;
      ageGrid[i] = isAlive ? 1 : 0;
      if (isAlive) aliveCount++;
    }
    generation = 0;
    updateTelemetry();
    render();
  }

  function clearGrid() {
    currentGrid.fill(0);
    nextGrid.fill(0);
    ageGrid.fill(0);
    generation = 0;
    aliveCount = 0;
    updateTelemetry();
    render();
  }

  /* ── Cellular Automaton Step (Toroidal & B3/S23) ────────────────────────── */
  function stepSimulation() {
    let newAlive = 0;
    const isToroidal = toroidal;

    for (let y = 0; y < N; y++) {
      const yUp = isToroidal ? ((y === 0 ? N - 1 : y - 1) * N) : (y > 0 ? (y - 1) * N : -1);
      const yCurr = y * N;
      const yDown = isToroidal ? ((y === N - 1 ? 0 : y + 1) * N) : (y < N - 1 ? (y + 1) * N : -1);

      for (let x = 0; x < N; x++) {
        const xLeft = isToroidal ? (x === 0 ? N - 1 : x - 1) : (x > 0 ? x - 1 : -1);
        const xRight = isToroidal ? (x === N - 1 ? 0 : x + 1) : (x < N - 1 ? x + 1 : -1);

        let neighbors = 0;

        if (yUp !== -1) {
          if (xLeft !== -1) neighbors += currentGrid[yUp + xLeft];
          neighbors += currentGrid[yUp + x];
          if (xRight !== -1) neighbors += currentGrid[yUp + xRight];
        }

        if (xLeft !== -1) neighbors += currentGrid[yCurr + xLeft];
        if (xRight !== -1) neighbors += currentGrid[yCurr + xRight];

        if (yDown !== -1) {
          if (xLeft !== -1) neighbors += currentGrid[yDown + xLeft];
          neighbors += currentGrid[yDown + x];
          if (xRight !== -1) neighbors += currentGrid[yDown + xRight];
        }

        const idx = yCurr + x;
        const cell = currentGrid[idx];
        let nextState = 0;

        if (cell === 1) {
          // Live cell survives with 2 or 3 neighbours
          nextState = (neighbors === 2 || neighbors === 3) ? 1 : 0;
        } else {
          // Dead cell reproduces with exactly 3 neighbours
          nextState = (neighbors === 3) ? 1 : 0;
        }

        nextGrid[idx] = nextState;

        if (nextState === 1) {
          newAlive++;
          ageGrid[idx] = cell === 1 ? Math.min(65535, ageGrid[idx] + 1) : 1;
        } else {
          ageGrid[idx] = 0;
        }
      }
    }

    // Pointer swap
    const temp = currentGrid;
    currentGrid = nextGrid;
    nextGrid = temp;

    aliveCount = newAlive;
    generation++;
    genCount++;
  }

  /* ── Direct Pixel Rendering to Canvas ──────────────────────────────────── */
  function render() {
    const palette = THEME_PALETTES[activeTheme] || THEME_PALETTES['theme-emerald'];
    const bgPixel = palette.bg;
    const youngPixel = palette.young;
    const maturePixel = palette.mature;
    const oldPixel = palette.old;
    const ancientPixel = palette.ancient;
    const useAging = cellAging;

    const totalCells = N * N;
    for (let i = 0; i < totalCells; i++) {
      if (currentGrid[i] === 1) {
        if (!useAging) {
          offPixelBuf32[i] = maturePixel;
        } else {
          const age = ageGrid[i];
          if (age < 3) {
            offPixelBuf32[i] = youngPixel;
          } else if (age < 12) {
            offPixelBuf32[i] = maturePixel;
          } else if (age < 35) {
            offPixelBuf32[i] = oldPixel;
          } else {
            offPixelBuf32[i] = ancientPixel;
          }
        }
      } else {
        offPixelBuf32[i] = bgPixel;
      }
    }

    offCtx.putImageData(offImageData, 0, 0);

    // Draw offscreen image onto main viewport canvas with Pan & Zoom
    ctx.save();
    ctx.imageSmoothingEnabled = zoom < 2.0; // crisp pixels when zoomed in
    ctx.fillStyle = '#030712';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.translate(panX, panY);
    ctx.scale(zoom, zoom);

    ctx.drawImage(offCanvas, 0, 0);

    // Grid lines when zoomed in sufficiently
    if (zoom >= 10.0) {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 1 / zoom;
      ctx.beginPath();
      for (let x = 0; x <= N; x++) {
        ctx.moveTo(x, 0);
        ctx.lineTo(x, N);
      }
      for (let y = 0; y <= N; y++) {
        ctx.moveTo(0, y);
        ctx.lineTo(N, y);
      }
      ctx.stroke();
    }

    ctx.restore();
  }

  /* ── Telemetry & Metrics Display ───────────────────────────────────────── */
  function updateTelemetry() {
    valGen.textContent = generation.toLocaleString();
    valAlive.textContent = aliveCount.toLocaleString();
    const pct = ((aliveCount / (N * N)) * 100).toFixed(2);
    valAlivePct.textContent = `(${pct}%)`;

    valSimSpeed.innerHTML = `${actualGenRate.toFixed(1)} <small>gen/s</small>`;
    valThroughput.innerHTML = `${throughput.toFixed(2)} <small>M cells/s</small>`;
    valFps.innerHTML = `${Math.round(fps)} <small>FPS</small>`;
  }

  /* ── Animation & Simulation Loop ───────────────────────────────────────── */
  function mainLoop(now) {
    const deltaMs = now - lastFrameTime;
    lastFrameTime = now;
    frameCount++;

    // Calculate FPS every 500ms
    if (now - lastGenTime >= 500) {
      const elapsedSec = (now - lastGenTime) / 1000;
      fps = (frameCount / elapsedSec);
      actualGenRate = (genCount / elapsedSec);
      throughput = (genCount * (N * N)) / (elapsedSec * 1e6);

      frameCount = 0;
      genCount = 0;
      lastGenTime = now;
      updateTelemetry();
    }

    // Step simulation based on target rate
    if (isRunning) {
      const stepInterval = 1000 / targetGenPerSec;
      genAccumulator += deltaMs;

      // Prevent spiral of death if tab was inactive
      if (genAccumulator > 500) genAccumulator = stepInterval;

      while (genAccumulator >= stepInterval) {
        stepSimulation();
        genAccumulator -= stepInterval;
      }
    }

    render();
    requestAnimationFrame(mainLoop);
  }

  /* ── Coordinate Transformation Helpers ─────────────────────────────────── */
  function screenToGrid(screenX, screenY) {
    const rect = canvas.getBoundingClientRect();
    const clientX = screenX - rect.left;
    const clientY = screenY - rect.top;

    const gridX = Math.floor((clientX - panX) / zoom);
    const gridY = Math.floor((clientY - panY) / zoom);

    return { x: gridX, y: gridY };
  }

  function setCell(x, y, state) {
    if (x >= 0 && x < N && y >= 0 && y < N) {
      const idx = y * N + x;
      if (currentGrid[idx] !== state) {
        currentGrid[idx] = state;
        ageGrid[idx] = state ? 1 : 0;
        aliveCount += state ? 1 : -1;
      }
    }
  }

  /* ── Pattern Presets Library ───────────────────────────────────────────── */
  const PRESETS = {
    gosper: [
      [24,0],[22,1],[24,1],[12,2],[13,2],[20,2],[21,2],[34,2],[35,2],[11,3],[15,3],[20,3],[21,3],[34,3],[35,3],
      [0,4],[1,4],[10,4],[16,4],[20,4],[21,4],[0,5],[1,5],[10,5],[14,5],[16,5],[17,5],[22,5],[24,5],[10,6],
      [16,6],[24,6],[11,7],[15,7],[12,8],[13,8]
    ],
    simkin: [
      [0,0],[1,0],[0,1],[1,1],[4,4],[5,4],[4,5],[5,5],[7,0],[8,0],[10,1],[11,2],[11,3],[10,4],[7,4],[21,0],
      [22,0],[21,1],[23,1],[24,2],[24,3],[25,3],[30,7],[31,7],[30,8],[32,8],[30,9],[31,9]
    ],
    pulsar: [
      [2,0],[3,0],[4,0],[8,0],[9,0],[10,0],
      [0,2],[5,2],[7,2],[12,2],[0,3],[5,3],[7,3],[12,3],[0,4],[5,4],[7,4],[12,4],
      [2,5],[3,5],[4,5],[8,5],[9,5],[10,5],
      [2,7],[3,7],[4,7],[8,7],[9,7],[10,7],
      [0,8],[5,8],[7,8],[12,8],[0,9],[5,9],[7,9],[12,9],[0,10],[5,10],[7,10],[12,10],
      [2,12],[3,12],[4,12],[8,12],[9,12],[10,12]
    ],
    pentadecathlon: [
      [1,0],[2,0],[0,1],[3,1],[0,2],[3,2],[1,3],[2,3],[1,4],[2,4],[1,5],[2,5],[1,6],[2,6],
      [0,7],[3,7],[0,8],[3,8],[1,9],[2,9]
    ],
    lwss: [
      [1,0],[4,0],[0,1],[0,2],[4,2],[0,3],[1,3],[2,3],[3,3]
    ],
    acorn: [
      [1,0],[3,1],[0,2],[1,2],[4,2],[5,2],[6,2]
    ],
    diehard: [
      [6,0],[0,1],[1,1],[1,2],[5,2],[6,2],[7,2]
    ]
  };

  function stampPreset(presetKey) {
    if (presetKey === 'random') {
      randomizeGrid(30);
      return;
    }
    const pattern = PRESETS[presetKey];
    if (!pattern) return;

    clearGrid();

    // Center pattern on grid
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    pattern.forEach(([px, py]) => {
      if (px < minX) minX = px;
      if (px > maxX) maxX = px;
      if (py < minY) minY = py;
      if (py > maxY) maxY = py;
    });

    const pWidth = maxX - minX + 1;
    const pHeight = maxY - minY + 1;
    const offsetX = Math.floor((N - pWidth) / 2) - minX;
    const offsetY = Math.floor((N - pHeight) / 2) - minY;

    pattern.forEach(([px, py]) => {
      const gx = offsetX + px;
      const gy = offsetY + py;
      if (gx >= 0 && gx < N && gy >= 0 && gy < N) {
        currentGrid[gy * N + gx] = 1;
        ageGrid[gy * N + gx] = 1;
        aliveCount++;
      }
    });

    generation = 0;
    updateTelemetry();
    render();
  }

  /* ── Resize Handler ────────────────────────────────────────────────────── */
  function resizeCanvas() {
    canvas.width = viewport.clientWidth;
    canvas.height = viewport.clientHeight;
    render();
  }

  /* ── Event Listeners ───────────────────────────────────────────────────── */
  window.addEventListener('resize', resizeCanvas);

  /* Play / Pause Toggle */
  function togglePlayPause() {
    isRunning = !isRunning;
    if (isRunning) {
      iconPlay.style.display = 'none';
      iconPause.style.display = 'block';
      labelPlayPause.textContent = 'Pause';
      btnPlayPause.classList.add('active');
    } else {
      iconPlay.style.display = 'block';
      iconPause.style.display = 'none';
      labelPlayPause.textContent = 'Play';
      btnPlayPause.classList.remove('active');
    }
  }

  btnPlayPause.addEventListener('click', togglePlayPause);

  btnStep.addEventListener('click', () => {
    if (!isRunning) {
      stepSimulation();
      updateTelemetry();
      render();
    }
  });

  btnRandomize.addEventListener('click', () => randomizeGrid(30));
  btnClear.addEventListener('click', clearGrid);
  btnResetView.addEventListener('click', () => {
    resetView();
    render();
  });

  btnFullscreen.addEventListener('click', () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen().catch(() => {});
    }
  });

  /* Speed Controls */
  sliderSpeed.addEventListener('input', (e) => {
    targetGenPerSec = parseInt(e.target.value, 10);
    labelSpeedVal.textContent = `${targetGenPerSec} gen/s`;
    document.querySelectorAll('.btn-chip').forEach(c => c.classList.remove('active'));
  });

  document.querySelectorAll('.btn-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.btn-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      const val = parseInt(chip.dataset.speed, 10);
      sliderSpeed.value = val;
      targetGenPerSec = val;
      labelSpeedVal.textContent = `${val} gen/s`;
    });
  });

  /* Grid Resolution Change */
  selectGridSize.addEventListener('change', (e) => {
    const newSize = parseInt(e.target.value, 10);
    initGrid(newSize);
  });

  /* Preset Selection */
  selectPreset.addEventListener('change', (e) => {
    stampPreset(e.target.value);
  });

  /* Theme Selection */
  selectTheme.addEventListener('change', (e) => {
    document.body.className = e.target.value;
    activeTheme = e.target.value;
    render();
  });

  /* Toggles */
  checkAging.addEventListener('change', (e) => {
    cellAging = e.target.checked;
    render();
  });

  checkToroidal.addEventListener('change', (e) => {
    toroidal = e.target.checked;
  });

  /* Tool Selection */
  function setActiveTool(tool) {
    activeTool = tool;
    toolDraw.classList.toggle('active', tool === 'draw');
    toolErase.classList.toggle('active', tool === 'erase');
    toolPan.classList.toggle('active', tool === 'pan');

    viewport.style.cursor = tool === 'pan' ? 'grab' : 'crosshair';
  }

  toolDraw.addEventListener('click', () => setActiveTool('draw'));
  toolErase.addEventListener('click', () => setActiveTool('erase'));
  toolPan.addEventListener('click', () => setActiveTool('pan'));

  /* ── Interactive Canvas Mouse & Touch Handling ─────────────────────────── */
  viewport.addEventListener('mousedown', (e) => {
    // Middle click or Pan tool initiates drag
    if (e.button === 1 || activeTool === 'pan' || e.shiftKey) {
      isDragging = true;
      dragStartX = e.clientX - panX;
      dragStartY = e.clientY - panY;
      viewport.style.cursor = 'grabbing';
      return;
    }

    if (e.button === 0) { // Left click
      isDrawingOnMove = true;
      const { x, y } = screenToGrid(e.clientX, e.clientY);
      const state = (activeTool === 'erase' || e.altKey) ? 0 : 1;
      setCell(x, y, state);
      updateTelemetry();
      render();
    }
  });

  window.addEventListener('mousemove', (e) => {
    if (isDragging) {
      panX = e.clientX - dragStartX;
      panY = e.clientY - dragStartY;
      render();
      return;
    }

    // Inspect hovered cell
    const { x, y } = screenToGrid(e.clientX, e.clientY);
    if (x >= 0 && x < N && y >= 0 && y < N) {
      hudCoords.textContent = `Pos: (${x}, ${y})`;
      const cellState = currentGrid[y * N + x];
      const age = ageGrid[y * N + x];
      hudHoverState.textContent = cellState ? `Alive (Age: ${age})` : `Dead`;
    } else {
      hudCoords.textContent = `Pos: (—, —)`;
      hudHoverState.textContent = `Outside Grid`;
    }

    // Drawing while dragging
    if (isDrawingOnMove) {
      const state = (activeTool === 'erase' || e.altKey) ? 0 : 1;
      setCell(x, y, state);
      updateTelemetry();
      render();
    }
  });

  window.addEventListener('mouseup', () => {
    if (isDragging) {
      isDragging = false;
      viewport.style.cursor = activeTool === 'pan' ? 'grab' : 'crosshair';
    }
    isDrawingOnMove = false;
  });

  /* Mouse Wheel Zoom (Centered at Cursor) */
  viewport.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const zoomFactor = e.deltaY < 0 ? 1.2 : 0.833;
    const newZoom = Math.min(Math.max(zoom * zoomFactor, 0.05), 50.0);

    // Keep world position under mouse fixed
    panX = mouseX - (mouseX - panX) * (newZoom / zoom);
    panY = mouseY - (mouseY - panY) * (newZoom / zoom);
    zoom = newZoom;

    updateZoomHUD();
    render();
  }, { passive: false });

  /* ── Keyboard Shortcuts ────────────────────────────────────────────────── */
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

    switch (e.code) {
      case 'Space':
        e.preventDefault();
        togglePlayPause();
        break;
      case 'KeyS':
        if (!isRunning) {
          stepSimulation();
          updateTelemetry();
          render();
        }
        break;
      case 'KeyR':
        randomizeGrid(30);
        break;
      case 'KeyC':
        clearGrid();
        break;
      case 'KeyP':
        setActiveTool(activeTool === 'pan' ? 'draw' : 'pan');
        break;
      case 'KeyE':
        setActiveTool('erase');
        break;
      case 'KeyD':
        setActiveTool('draw');
        break;
    }
  });

  /* ── Benchmark Modal ───────────────────────────────────────────────────── */
  btnToggleBenchmark.addEventListener('click', () => {
    modalBenchmark.style.display = 'flex';
  });

  btnCloseBenchmark.addEventListener('click', () => {
    modalBenchmark.style.display = 'none';
  });

  modalBenchmark.addEventListener('click', (e) => {
    if (e.target === modalBenchmark) {
      modalBenchmark.style.display = 'none';
    }
  });

  /* ── Launch Application ────────────────────────────────────────────────── */
  resizeCanvas();
  initGrid(1024);
  requestAnimationFrame(mainLoop);

})();
