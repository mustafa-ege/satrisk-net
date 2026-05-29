from pathlib import Path
import os
import json
import rasterio
from rasterio.warp import transform_bounds

def main():
    root = Path(os.getenv("SATRISK_DATASET_ROOT"))
    public_root = root.parent.parent / "frontend" / "public"
    
    regions = []
    seen = set()
    
    # scan for *_S2L2A.tif files
    for p in sorted(root.rglob("*_S2L2A.tif")):
        region_id = p.parent.name
        if region_id in seen:
            continue
            
        try:
            with rasterio.open(p) as src:
                crs = src.crs
                bounds = src.bounds
                
            left, bottom, right, top = transform_bounds(
                crs, "EPSG:4326", bounds.left, bounds.bottom, bounds.right, bounds.top
            )
            bbox = [left, bottom, right, top]
            
            regions.append({
                "id": region_id,
                "bbox": bbox
            })
            seen.add(region_id)
            print(f"Scanned region: {region_id} with bbox {bbox}")
        except Exception as e:
            print(f"Error scanning {p}: {e}")
            
    regions.sort(key=lambda x: x["id"])
    
    out_path = public_root / "data" / "regions.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(regions, f, indent=2)
        
    print(f"Successfully generated regions.json at {out_path} with {len(regions)} regions.")

if __name__ == "__main__":
    main()
