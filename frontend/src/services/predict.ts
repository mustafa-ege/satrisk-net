/** Matches backend `PredictRequest` / `PredictResponse`. */

export type PredictModel = 'unet' | 'deeplab' | 'segformer'

export type PredictRequestBody = {
  region_id: string
  model: PredictModel
}

export type PredictResponse = {
  region_id: string
  mask_url: string
  ground_truth_url: string
  image_url?: string
  metrics: {
    iou: number
  }
}

const PREDICT_URL = '/api/predict'

async function fetchStaticPredict(
  regionId: string,
  model: string,
): Promise<PredictResponse> {
  const res = await fetch('/data/metrics.json')
  if (!res.ok) {
    throw new Error(`Failed to load static metrics.json (${res.status})`)
  }
  const metrics = await res.json() as Array<{
    region_id: string
    model_name: string
    iou: number
    image_path: string
    prediction_path: string
    ground_truth_path: string
  }>

  const match = metrics.find(
    (m) => m.region_id === regionId && m.model_name === model,
  )

  if (!match) {
    throw new Error(`No precomputed metrics found for region_id=${regionId} model=${model}`)
  }

  return {
    region_id: match.region_id,
    mask_url: match.prediction_path,
    ground_truth_url: match.ground_truth_path,
    image_url: match.image_path,
    metrics: {
      iou: match.iou,
    },
  }
}

export async function postPredict(
  body: PredictRequestBody,
): Promise<PredictResponse> {
  const isStatic = import.meta.env.VITE_STATIC_MODE === 'true'
  if (isStatic) {
    return fetchStaticPredict(body.region_id, body.model)
  }

  try {
    const res = await fetch(PREDICT_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!res.ok) {
      const detail = await res.text()
      throw new Error(
        detail ? `Prediction failed (${res.status}): ${detail}` : `Prediction failed (${res.status})`,
      )
    }
    return await res.json() as PredictResponse
  } catch (err) {
    console.warn('Backend connection failed, falling back to static precomputed predictions...', err)
    return fetchStaticPredict(body.region_id, body.model)
  }
}

export const MODEL_OPTIONS: { value: PredictModel; label: string }[] = [
  { value: 'unet', label: 'U-Net' },
  { value: 'deeplab', label: 'DeepLabV3+' },
  { value: 'segformer', label: 'SegFormer' },
]
