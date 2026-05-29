/** Matches backend `DatasetRegionSummary`: bbox is (min_lon, min_lat, max_lon, max_lat). */
export type DatasetRegionSummary = {
  id: string
  bbox: [number, number, number, number]
}

/** Backend mounts the API under `/api` (see FastAPI `include_router(..., prefix="/api")`). */
const DATASETS_URL = '/api/datasets'

export async function fetchDatasetRegions(): Promise<DatasetRegionSummary[]> {
  const isStatic = import.meta.env.VITE_STATIC_MODE === 'true'
  if (isStatic) {
    const res = await fetch('/data/regions.json')
    if (!res.ok) {
      throw new Error(`Failed to load static regions (${res.status})`)
    }
    return res.json() as Promise<DatasetRegionSummary[]>
  }

  try {
    const res = await fetch(DATASETS_URL)
    if (!res.ok) {
      throw new Error(`Failed to load datasets (${res.status})`)
    }
    return await res.json() as DatasetRegionSummary[]
  } catch (err) {
    console.warn('Backend connection failed, falling back to static dataset regions...', err)
    const res = await fetch('/data/regions.json')
    if (!res.ok) {
      throw new Error(`Failed to load fallback static regions (${res.status})`)
    }
    return res.json() as Promise<DatasetRegionSummary[]>
  }
}
