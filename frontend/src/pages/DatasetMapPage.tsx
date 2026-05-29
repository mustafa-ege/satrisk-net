import { useEffect, useMemo, useState } from 'react'
import { DatasetMap } from '../components/DatasetMap'
import { PredictionPanel } from '../components/PredictionPanel'
import {
  fetchDatasetRegions,
  type DatasetRegionSummary,
} from '../services/datasets'
import {
  type PredictModel,
  postPredict,
} from '../services/predict'
import { buildRasterMaskDataUrl } from '../utils/rasterMask'

import styles from './DatasetMapPage.module.css'

export function DatasetMapPage() {
  const [regions, setRegions] = useState<DatasetRegionSummary[]>([])
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)

  const [basemap, setBasemap] = useState<'streets' | 'satellite'>('streets')
  const [model, setModel] = useState<PredictModel>('unet')
  const [showGroundTruth, setShowGroundTruth] = useState(true)
  const [iou, setIou] = useState<number | null>(null)
  const [predictionOverlayUrl, setPredictionOverlayUrl] = useState<
    string | null
  >(null)
  const [groundTruthOverlayUrl, setGroundTruthOverlayUrl] = useState<
    string | null
  >(null)
  const [predictLoading, setPredictLoading] = useState(false)
  const [predictError, setPredictError] = useState<string | null>(null)
  const [panelPredUrl, setPanelPredUrl] = useState<string | null>(null)
  const [panelGtUrl, setPanelGtUrl] = useState<string | null>(null)
  const [panelImageUrl, setPanelImageUrl] = useState<string | null>(null)
  const [regionImageVersion, setRegionImageVersion] = useState(0)

  const selectedRegion = useMemo(
    () => regions.find((r) => r.id === selectedRegionId) ?? null,
    [regions, selectedRegionId],
  )

  const panelIou = selectedRegionId ? iou : null
  const panelLoading = !!selectedRegionId && predictLoading
  const panelError = selectedRegionId ? predictError : null
  const mapPredUrl = selectedRegionId ? predictionOverlayUrl : null
  const mapGtUrl = selectedRegionId ? groundTruthOverlayUrl : null

  useEffect(() => {
    let cancelled = false
    fetchDatasetRegions()
      .then((data) => {
        if (!cancelled) setRegions(data)
      })
      .catch(() => {
        if (!cancelled) setLoadError('Could not load dataset regions.')
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!selectedRegionId) {
      return
    }

    let cancelled = false
    const regionId = selectedRegionId

    const run = async () => {
      setPredictLoading(true)
      setPredictError(null)
      setIou(null)
      setPredictionOverlayUrl(null)
      setGroundTruthOverlayUrl(null)
      setPanelPredUrl(null)
      setPanelGtUrl(null)
      setPanelImageUrl(null)
      setRegionImageVersion((v) => v + 1)

      try {
        const res = await postPredict({ region_id: regionId, model })
        setPanelPredUrl(res.mask_url)
        setPanelGtUrl(res.ground_truth_url)
        setPanelImageUrl(res.image_url ?? `/static/region_images/${encodeURIComponent(regionId)}.png`)
        // Backend writes the quicklook during /predict. Force the <img> to retry by
        // changing the query param after the request completes.
        setRegionImageVersion((v) => v + 1)
        const [predUrl, gtUrl] = await Promise.all([
          buildRasterMaskDataUrl(
            res.mask_url,
            'prediction',
            `${model}|${res.region_id}`,
          ),
          buildRasterMaskDataUrl(
            res.ground_truth_url,
            'ground_truth',
            res.region_id,
          ),
        ])
        if (cancelled) return
        setIou(res.metrics.iou)
        setPredictionOverlayUrl(predUrl)
        setGroundTruthOverlayUrl(gtUrl)
      } catch (err: unknown) {
        if (cancelled) return
        setPredictError(
          err instanceof Error ? err.message : 'Prediction request failed.',
        )
      } finally {
        if (!cancelled) setPredictLoading(false)
      }
    }

    void run()

    return () => {
      cancelled = true
    }
  }, [selectedRegionId, model])

  return (
    <div className={styles.page}>
      {loadError ? (
        <p className={styles.banner} role="status">
          {loadError}
        </p>
      ) : null}
      <PredictionPanel
        model={model}
        onModelChange={setModel}
        showGroundTruth={showGroundTruth}
        onShowGroundTruthChange={setShowGroundTruth}
        iou={panelIou}
        loading={panelLoading}
        error={panelError}
        regionId={selectedRegionId}
        regionImageUrl={
          selectedRegionId && panelImageUrl
            ? `${panelImageUrl}?v=${regionImageVersion}`
            : null
        }
        predictionMaskUrl={panelPredUrl}
        groundTruthMaskUrl={panelGtUrl}
      />
      <DatasetMap
        regions={regions}
        selectedRegionId={selectedRegionId}
        onSelectRegion={setSelectedRegionId}
        overlayBbox={selectedRegion?.bbox ?? null}
        predictionOverlayUrl={mapPredUrl}
        groundTruthOverlayUrl={mapGtUrl}
        showGroundTruthOverlay={showGroundTruth}
        basemap={basemap}
        onBasemapChange={setBasemap}
      />
    </div>
  )
}
