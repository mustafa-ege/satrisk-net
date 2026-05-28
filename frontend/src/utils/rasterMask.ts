/**
 * Builds a PNG data URL for map overlay. Backend mask URLs may point to text
 * placeholders; we still fetch them to seed the pattern so overlays stay stable
 * per region/model.
 */
export type MaskKind = 'prediction' | 'ground_truth'

function hashSeed(s: string): number {
  let h = 2166136261
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

function mulberry32(seed: number) {
  return function next() {
    let t = (seed += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const SIZE = 256

async function tryBuildOverlayFromPng(
  maskUrl: string,
  kind: MaskKind,
): Promise<string | null> {
  let blob: Blob | null = null
  let contentType = ''
  try {
    const res = await fetch(maskUrl)
    if (!res.ok) return null
    contentType = res.headers.get('content-type') ?? ''
    blob = await res.blob()
  } catch {
    return null
  }

  const isPng =
    contentType.includes('image/png') ||
    maskUrl.toLowerCase().includes('.png')
  if (!isPng || !blob) return null

  let bmp: ImageBitmap
  try {
    bmp = await createImageBitmap(blob)
  } catch {
    return null
  }

  const canvas = document.createElement('canvas')
  canvas.width = bmp.width
  canvas.height = bmp.height
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  if (!ctx) return null

  ctx.drawImage(bmp, 0, 0)
  const img = ctx.getImageData(0, 0, canvas.width, canvas.height)
  const data = img.data

  const base =
    kind === 'prediction'
      ? { r: 210, g: 18, b: 18 } // strong fire red
      : { r: 59, g: 130, b: 246 } // blue

  // Turn the binary mask into a tinted RGBA overlay.
  // We assume mask pixels are either black or white.
  for (let i = 0; i < data.length; i += 4) {
    const v = data[i] // red channel (grayscale so r=g=b)
    if (v > 16) {
      data[i] = base.r
      data[i + 1] = base.g
      data[i + 2] = base.b
      data[i + 3] = kind === 'prediction' ? 235 : 150
    } else {
      data[i + 3] = 0
    }
  }

  ctx.putImageData(img, 0, 0)
  return canvas.toDataURL('image/png')
}

export async function buildRasterMaskDataUrl(
  maskUrl: string,
  kind: MaskKind,
  extraSeed: string,
): Promise<string> {
  const overlay = await tryBuildOverlayFromPng(maskUrl, kind)
  if (overlay) return overlay

  let textSeed = ''
  try {
    const res = await fetch(maskUrl)
    if (res.ok) {
      textSeed = await res.text()
    }
  } catch {
    /* ignore — still render a deterministic overlay */
  }

  const seedStr = `${textSeed}|${extraSeed}|${kind}`
  const rng = mulberry32(hashSeed(seedStr))

  const canvas = document.createElement('canvas')
  canvas.width = SIZE
  canvas.height = SIZE
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    return 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=='
  }

  const base =
    kind === 'prediction'
      ? { r: 210, g: 18, b: 18 }
      : { r: 59, g: 130, b: 246 }

  ctx.fillStyle =
    kind === 'prediction'
      ? `rgba(${base.r}, ${base.g}, ${base.b}, 0.52)`
      : `rgba(${base.r}, ${base.g}, ${base.b}, 0.28)`
  ctx.fillRect(0, 0, SIZE, SIZE)

  const stripes = kind === 'prediction' ? 14 : 18
  ctx.globalAlpha = kind === 'prediction' ? 0.38 : 0.22
  for (let i = 0; i < stripes; i++) {
    const offset = (i / stripes) * (SIZE + 40) - 20 + rng() * 8
    ctx.fillStyle = i % 2 === 0 ? 'rgba(255,255,255,0.9)' : 'rgba(0,0,0,0.25)'
    ctx.beginPath()
    if (kind === 'prediction') {
      ctx.rect(offset, 0, 6 + rng() * 6, SIZE)
    } else {
      ctx.rect(0, offset, SIZE, 6 + rng() * 6)
    }
    ctx.fill()
  }

  ctx.globalAlpha = 0.35
  for (let n = 0; n < 55; n++) {
    const x = rng() * SIZE
    const y = rng() * SIZE
    const w = 4 + rng() * 28
    const h = 4 + rng() * 28
    ctx.fillStyle =
      rng() > 0.5
        ? `rgba(${Math.min(255, base.r + 40)}, ${Math.min(255, base.g + 40)}, ${Math.min(255, base.b + 40)}, 0.5)`
        : 'rgba(0,0,0,0.2)'
    ctx.fillRect(x, y, w, h)
  }

  ctx.globalAlpha = 1
  return canvas.toDataURL('image/png')
}
