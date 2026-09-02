import { useEffect, useRef } from 'react';

interface Rect {
  normX: number;
  normY: number;
  normW: number;
  normH: number;
}

interface ImageBox {
  width: number;
  height: number;
  offsetX: number;
  offsetY: number;
}

interface TransformState {
  scale: number;
  positionX: number;
  positionY: number;
}

interface WebglTexturedCanvasProps {
  src: string | null;
  rect?: Rect;
  imageRenderSize: ImageBox;
  transformState: TransformState;
  isMaxZoom: boolean;
  style?: React.CSSProperties;
  className?: string;
  onLoadingChange?: (isLoading: boolean) => void;
}

const FULL_RECT: Rect = { normX: 0, normY: 0, normW: 1, normH: 1 };

const VERTEX_SHADER_SOURCE = `
  attribute vec2 aPos;
  attribute vec2 aUv;
  varying vec2 vUv;
  uniform vec2 uScale;
  uniform vec2 uTranslate;
  void main() {
    gl_Position = vec4(aPos * uScale + uTranslate, 0.0, 1.0);
    vUv = aUv;
  }
`;

const FRAGMENT_SHADER_SOURCE = `
  precision mediump float;
  varying vec2 vUv;
  uniform sampler2D uTex;
  void main() {
    gl_FragColor = texture2D(uTex, vUv);
  }
`;

function compileShader(gl: WebGLRenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error('Failed to create shader');
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(info || 'Shader compile failed');
  }
  return shader;
}

// Renders a single image texture into a full-viewport canvas, applying the same
// pan/zoom transform the app applies via CSS elsewhere - but entirely inside the
// shader (uScale/uTranslate), not via a CSS `transform: scale()` on the canvas
// itself. This is deliberate: scaling an already-rasterized layer via CSS leaves
// the browser's compositor to pick a sampling filter for the blow-up, and
// WebKitGTK does not reliably honor `image-rendering: pixelated` through that
// path. Doing the scale in the shader means WE choose NEAREST vs LINEAR
// filtering directly against the source texture, with no compositor step in
// between that could silently reintroduce smoothing.
export default function WebglTexturedCanvas({
  src,
  rect = FULL_RECT,
  imageRenderSize,
  transformState,
  isMaxZoom,
  style,
  className,
  onLoadingChange,
}: WebglTexturedCanvasProps) {
  const onLoadingChangeRef = useRef(onLoadingChange);
  onLoadingChangeRef.current = onLoadingChange;
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const glRef = useRef<WebGLRenderingContext | null>(null);
  const programRef = useRef<WebGLProgram | null>(null);
  const textureRef = useRef<WebGLTexture | null>(null);
  const uniformsRef = useRef<{
    uScale: WebGLUniformLocation;
    uTranslate: WebGLUniformLocation;
  } | null>(null);
  const loadedSrcRef = useRef<string | null>(null);
  const hasImageRef = useRef(false);

  const rectRef = useRef(rect);
  rectRef.current = rect;
  const imageRenderSizeRef = useRef(imageRenderSize);
  imageRenderSizeRef.current = imageRenderSize;
  const transformStateRef = useRef(transformState);
  transformStateRef.current = transformState;
  const isMaxZoomRef = useRef(isMaxZoom);
  isMaxZoomRef.current = isMaxZoom;

  const drawRef = useRef<() => void>(() => {});

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = (canvas.getContext('webgl', { premultipliedAlpha: false, alpha: true }) ||
      canvas.getContext('experimental-webgl', {
        premultipliedAlpha: false,
        alpha: true,
      })) as WebGLRenderingContext | null;
    if (!gl) {
      console.error('WebglTexturedCanvas: WebGL is not available');
      return;
    }
    glRef.current = gl;

    const setup = () => {
      const vertexShader = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER_SOURCE);
      const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER_SOURCE);
      const program = gl.createProgram();
      if (!program) throw new Error('Failed to create program');
      gl.attachShader(program, vertexShader);
      gl.attachShader(program, fragmentShader);
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        throw new Error(gl.getProgramInfoLog(program) || 'Program link failed');
      }
      gl.useProgram(program);
      programRef.current = program;

      const verts = new Float32Array([-1, -1, 0, 1, 1, -1, 1, 1, -1, 1, 0, 0, 1, 1, 1, 0]);
      const vbo = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);

      const aPos = gl.getAttribLocation(program, 'aPos');
      const aUv = gl.getAttribLocation(program, 'aUv');
      gl.enableVertexAttribArray(aPos);
      gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 16, 0);
      gl.enableVertexAttribArray(aUv);
      gl.vertexAttribPointer(aUv, 2, gl.FLOAT, false, 16, 8);

      const uScale = gl.getUniformLocation(program, 'uScale');
      const uTranslate = gl.getUniformLocation(program, 'uTranslate');
      const uTex = gl.getUniformLocation(program, 'uTex');
      if (!uScale || !uTranslate) throw new Error('Failed to locate uniforms');
      gl.uniform1i(uTex, 0);
      uniformsRef.current = { uScale, uTranslate };

      const texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      textureRef.current = texture;

      gl.clearColor(0, 0, 0, 0);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    };

    try {
      setup();
    } catch (err) {
      console.error('WebglTexturedCanvas: setup failed', err);
      return;
    }

    hasImageRef.current = false;
    loadedSrcRef.current = null;

    const draw = () => {
      const canvasEl = canvasRef.current;
      const context = glRef.current;
      const texture = textureRef.current;
      const uniforms = uniformsRef.current;
      if (!canvasEl || !context || !texture || !uniforms) return;

      const dpr = window.devicePixelRatio || 1;
      const parent = canvasEl.parentElement;
      const cw = Math.max(1, Math.round((parent?.clientWidth || canvasEl.clientWidth || 1) * dpr));
      const ch = Math.max(1, Math.round((parent?.clientHeight || canvasEl.clientHeight || 1) * dpr));
      if (canvasEl.width !== cw || canvasEl.height !== ch) {
        canvasEl.width = cw;
        canvasEl.height = ch;
      }
      context.viewport(0, 0, cw, ch);
      context.clear(context.COLOR_BUFFER_BIT);

      if (!hasImageRef.current) return;

      const irs = imageRenderSizeRef.current;
      const ts = transformStateRef.current;
      const r = rectRef.current;

      const screenLeft = ts.positionX + (irs.offsetX + r.normX * irs.width) * ts.scale;
      const screenTop = ts.positionY + (irs.offsetY + r.normY * irs.height) * ts.scale;
      const screenW = r.normW * irs.width * ts.scale;
      const screenH = r.normH * irs.height * ts.scale;

      const clipLeft = ((screenLeft * dpr) / cw) * 2 - 1;
      const clipRight = (((screenLeft + screenW) * dpr) / cw) * 2 - 1;
      const clipTop = 1 - ((screenTop * dpr) / ch) * 2;
      const clipBottom = 1 - (((screenTop + screenH) * dpr) / ch) * 2;

      const sx = (clipRight - clipLeft) / 2;
      const sy = (clipTop - clipBottom) / 2;
      const tx = (clipRight + clipLeft) / 2;
      const ty = (clipTop + clipBottom) / 2;

      context.bindTexture(context.TEXTURE_2D, texture);
      const filter = isMaxZoomRef.current ? context.NEAREST : context.LINEAR;
      context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MIN_FILTER, filter);
      context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MAG_FILTER, filter);

      context.useProgram(programRef.current);
      context.uniform2f(uniforms.uScale, sx, sy);
      context.uniform2f(uniforms.uTranslate, tx, ty);
      context.drawArrays(context.TRIANGLE_STRIP, 0, 4);
    };

    drawRef.current = draw;
    draw();

    const handleContextLost = (e: Event) => {
      e.preventDefault();
      hasImageRef.current = false;
      loadedSrcRef.current = null;
    };
    const handleContextRestored = () => {
      try {
        setup();
        loadedSrcRef.current = null; // force the src-effect below to re-upload
      } catch (err) {
        console.error('WebglTexturedCanvas: failed to reinit after context restore', err);
      }
    };
    canvas.addEventListener('webglcontextlost', handleContextLost, false);
    canvas.addEventListener('webglcontextrestored', handleContextRestored, false);

    const resizeObserver = new ResizeObserver(() => draw());
    if (canvas.parentElement) resizeObserver.observe(canvas.parentElement);

    return () => {
      canvas.removeEventListener('webglcontextlost', handleContextLost);
      canvas.removeEventListener('webglcontextrestored', handleContextRestored);
      resizeObserver.disconnect();
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    if (!src) {
      hasImageRef.current = false;
      loadedSrcRef.current = null;
      drawRef.current();
      onLoadingChangeRef.current?.(false);
      return;
    }

    if (loadedSrcRef.current === src) return;

    onLoadingChangeRef.current?.(true);

    // Loaded via fetch()+createImageBitmap(), not a plain <img> element: a bitmap
    // built from bytes fetched in JS is never "CORS-tainted", whereas texImage2D
    // from a cross-origin/custom-scheme <img> (e.g. Tauri's asset:// thumbnails)
    // throws a SecurityError even though the browser displays that <img> just fine.
    // The app has no CSP configured, so fetch() works uniformly for blob:, asset://,
    // and any other scheme these sources use.
    (async () => {
      try {
        const response = await fetch(src);
        const blob = await response.blob();
        const bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none' });
        if (cancelled) {
          bitmap.close();
          return;
        }
        const gl = glRef.current;
        const texture = textureRef.current;
        if (!gl || !texture) {
          bitmap.close();
          return;
        }
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
        bitmap.close();
        hasImageRef.current = true;
        loadedSrcRef.current = src;
        drawRef.current();
        onLoadingChangeRef.current?.(false);
      } catch (err) {
        console.error('WebglTexturedCanvas: failed to load texture', src, err);
        onLoadingChangeRef.current?.(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [src]);

  useEffect(() => {
    drawRef.current();
  }, [
    transformState.scale,
    transformState.positionX,
    transformState.positionY,
    imageRenderSize.width,
    imageRenderSize.height,
    imageRenderSize.offsetX,
    imageRenderSize.offsetY,
    isMaxZoom,
    rect.normX,
    rect.normY,
    rect.normW,
    rect.normH,
  ]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', ...style }}
    />
  );
}
