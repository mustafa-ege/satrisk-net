import { useEffect } from 'react'
import type { LatLngBoundsExpression, LatLngExpression } from 'leaflet'
import L from 'leaflet'
import {
  ImageOverlay,
  MapContainer,
  Polygon,
  TileLayer,
  useMap,
} from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import type { DatasetRegionSummary } from '../services/datasets'

import styles from './DatasetMap.module.css'

function bboxToRectangleLatLngs(
  bbox: DatasetRegionSummary['bbox'],
): LatLngExpression[] {
  const [minLon, minLat, maxLon, maxLat] = bbox
  return [
    [minLat, minLon],
    [minLat, maxLon],
    [maxLat, maxLon],
    [maxLat, minLon],
  ]
}

function bboxToOverlayBounds(
  bbox: DatasetRegionSummary['bbox'],
): LatLngBoundsExpression {
  const [minLon, minLat, maxLon, maxLat] = bbox
  return [
    [minLat, minLon],
    [maxLat, maxLon],
  ]
}

function regionsToBounds(regions: DatasetRegionSummary[]): LatLngBoundsExpression | null {
  if (regions.length === 0) return null
  const bounds = L.latLngBounds([])
  for (const r of regions) {
    const [minLon, minLat, maxLon, maxLat] = r.bbox
    bounds.extend(L.latLngBounds(
      [minLat, minLon],
      [maxLat, maxLon],
    ))
  }
  return bounds
}

function FitBounds({ regions }: { regions: DatasetRegionSummary[] }) {
  const map = useMap()
  useEffect(() => {
    const bounds = regionsToBounds(regions)
    if (bounds) {
      map.fitBounds(bounds, { padding: [28, 28], maxZoom: 12 })
    }
  }, [map, regions])
  return null
}

type DatasetMapProps = {
  regions: DatasetRegionSummary[]
  selectedRegionId: string | null
  onSelectRegion: (id: string) => void
  overlayBbox: DatasetRegionSummary['bbox'] | null
  predictionOverlayUrl: string | null
  groundTruthOverlayUrl: string | null
  showGroundTruthOverlay: boolean
  basemap: 'streets' | 'satellite'
  onBasemapChange: (basemap: 'streets' | 'satellite') => void
}

const DEFAULT_CENTER: LatLngExpression = [37.1, 27.3]
const DEFAULT_ZOOM = 9

export function DatasetMap({
  regions,
  selectedRegionId,
  onSelectRegion,
  overlayBbox,
  predictionOverlayUrl,
  groundTruthOverlayUrl,
  showGroundTruthOverlay,
  basemap,
  onBasemapChange,
}: DatasetMapProps) {
  const maskBounds =
    overlayBbox !== null ? bboxToOverlayBounds(overlayBbox) : null

  const tile =
    basemap === 'satellite'
      ? {
          url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
          attribution:
            'Tiles &copy; Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community',
        }
      : {
          url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
          attribution:
            '&copy; <a href=\"https://www.openstreetmap.org/copyright\">OpenStreetMap</a>',
        }

  return (
    <div className={styles.wrapper}>
      <MapContainer
        className={styles.map}
        center={DEFAULT_CENTER}
        zoom={DEFAULT_ZOOM}
        scrollWheelZoom
      >
        <TileLayer attribution={tile.attribution} url={tile.url} />
        <FitBounds regions={regions} />
        {maskBounds && predictionOverlayUrl ? (
          <ImageOverlay
            url={predictionOverlayUrl}
            bounds={maskBounds}
            opacity={showGroundTruthOverlay ? 0.52 : 0.72}
            interactive={false}
          />
        ) : null}
        {maskBounds && showGroundTruthOverlay && groundTruthOverlayUrl ? (
          <ImageOverlay
            url={groundTruthOverlayUrl}
            bounds={maskBounds}
            opacity={0.42}
            interactive={false}
          />
        ) : null}
        {regions.map((region) => {
          const selected = region.id === selectedRegionId
          return (
            <Polygon
              key={region.id}
              positions={bboxToRectangleLatLngs(region.bbox)}
              pathOptions={{
                color: selected ? '#0284c7' : '#64748b',
                weight:selected ? 3 : 2,
                fillColor: selected ? '#0ea5e9' : '#94a3b8',
                fillOpacity: selected ? 0.22 : 0.22,
              }}
              eventHandlers={{
                click: () => {
                  onSelectRegion(region.id)
                },
              }}
            />
          )
        })}
      </MapContainer>

      <div className={styles.mapControls} aria-label="Map controls">
        <button
          type="button"
          className={styles.mapControlButton}
          onClick={() =>
            onBasemapChange(basemap === 'satellite' ? 'streets' : 'satellite')
          }
          aria-label="Toggle basemap"
          title="Toggle basemap"
        >
          {basemap === 'satellite' ? (
            <svg
              viewBox="0 0 24 24"
              width="18"
              height="18"
              aria-hidden="true"
              focusable="false"
            >
              <path
                fill="currentColor"
                d="M12 3a9 9 0 1 0 9 9a9.01 9.01 0 0 0-9-9m0 2a7 7 0 0 1 6.93 6H5.07A7 7 0 0 1 12 5m-7 8h14a7 7 0 0 1-14 0"
              />
            </svg>
          ) : (
            <svg
              viewBox="0 0 24 24"
              width="18"
              height="18"
              aria-hidden="true"
              focusable="false"
            >
              <path
                fill="currentColor"
                d="M3 6.5l7-3l7 3l4-1.7v12.4L17 19.5l-7-3l-7 3zM10 6.1L5 8.2v9.7l5-2.2zm2 0v9.6l5 2.2V8.2z"
              />
            </svg>
          )}
        </button>
      </div>
    </div>
  )
}
