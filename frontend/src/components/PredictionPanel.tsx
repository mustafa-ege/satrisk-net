import {
  MODEL_OPTIONS,
  type PredictModel,
} from '../services/predict'

import styles from './PredictionPanel.module.css'

type PredictionPanelProps = {
  model: PredictModel
  onModelChange: (model: PredictModel) => void
  showGroundTruth: boolean
  onShowGroundTruthChange: (show: boolean) => void
  iou: number | null
  loading: boolean
  error: string | null
  regionId: string | null
  regionImageUrl: string | null
  predictionMaskUrl: string | null
  groundTruthMaskUrl: string | null
}

export function PredictionPanel({
  model,
  onModelChange,
  showGroundTruth,
  onShowGroundTruthChange,
  iou,
  loading,
  error,
  regionId,
  regionImageUrl,
  predictionMaskUrl,
  groundTruthMaskUrl,
}: PredictionPanelProps) {
  return (
    <aside className={styles.panel} aria-label="Prediction and metrics">
      <div className={styles.headerRow}>
        <div className={styles.titleBlock}>
          <div className={styles.title}>Prediction</div>
          <div className={styles.subtitle}>
            {regionId ? `Region: ${regionId}` : 'Select a region'}
          </div>
        </div>
      </div>

      <label className={styles.field}>
        <span className={styles.label}>Model</span>
        <select
          className={styles.select}
          value={model}
          onChange={(e) => onModelChange(e.target.value as PredictModel)}
          disabled={loading}
        >
          {MODEL_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </label>

      <label className={styles.checkboxRow}>
        <input
          type="checkbox"
          checked={showGroundTruth}
          onChange={(e) => onShowGroundTruthChange(e.target.checked)}
        />
        <span>Show ground truth overlay</span>
      </label>

      <div className={styles.metrics}>
        <span className={styles.metricLabel}>IoU</span>
        <span className={styles.metricValue}>
          {loading && iou === null
            ? '…'
            : iou !== null
              ? iou.toFixed(4)
              : '—'}
        </span>
      </div>

      {error ? (
        <p className={styles.error} role="alert">
          {error}
        </p>
      ) : null}

      <div className={styles.previews} aria-label="Region previews">
        <section className={styles.previewSection}>
          <div className={styles.previewLabel}>Image + ground truth</div>
          <div className={styles.overlayFrame}>
            {regionImageUrl ? (
              <img className={styles.baseImage} src={regionImageUrl} alt="" />
            ) : (
              <div className={styles.placeholder}>No image</div>
            )}
            {groundTruthMaskUrl ? (
              <img
                className={styles.overlayImage}
                src={groundTruthMaskUrl}
                alt=""
              />
            ) : null}
          </div>
        </section>

        <div className={styles.maskPairRow}>
          <section className={styles.previewSection}>
            <div className={styles.previewLabel}>Prediction</div>
            <div className={styles.imageFrame}>
              {predictionMaskUrl ? (
                <img
                  className={styles.singleImage}
                  src={predictionMaskUrl}
                  alt=""
                />
              ) : (
                <div className={styles.placeholder}>—</div>
              )}
            </div>
          </section>
          <section className={styles.previewSection}>
            <div className={styles.previewLabel}>Ground truth</div>
            <div className={styles.imageFrame}>
              {groundTruthMaskUrl ? (
                <img
                  className={styles.singleImage}
                  src={groundTruthMaskUrl}
                  alt=""
                />
              ) : (
                <div className={styles.placeholder}>—</div>
              )}
            </div>
          </section>
        </div>
      </div>
    </aside>
  )
}
