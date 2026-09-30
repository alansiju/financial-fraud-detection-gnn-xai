"""api/explain.py - GNNExplainer endpoint"""
import asyncio
import torch
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from server.auth.dependencies import get_current_user
from server.ml import loader
from server.ml.explainer import explain_node
from server.config import STATIC_DIR

router = APIRouter(prefix="/api/explain", tags=["Explainability"])


@router.get("/{identifier}")
async def get_explanation(identifier: int, current_user=Depends(get_current_user)):
    if not loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    data = loader.get_data()

    node_id = loader.resolve_node_id(identifier)
    if node_id < 0 or node_id >= data.num_nodes:
        raise HTTPException(status_code=404, detail=f"Transaction or node {identifier} not found")

    tx_id = loader.get_tx_id_by_node(node_id) or identifier
    ei = data.edge_index
    nbr_count = int(torch.sum((ei[0] == node_id) | (ei[1] == node_id)).item())
    feat_shape = list(data.x[node_id].shape)

    print(f"[XAI] Request received")
    print(f"[XAI] TransactionID: {tx_id}")
    print(f"[XAI] Node index: {node_id}")
    print(f"[XAI] Neighbor count: {nbr_count}")
    print(f"[XAI] Feature shape: {feat_shape}")

    try:
        # explain_node is CPU-intensive (GNNExplainer 100 epochs) — run in thread
        # executor so the async event loop is not blocked.
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, explain_node, node_id)
        result["transaction_id"] = tx_id
        result["requested_id"] = identifier
        print(f"[XAI] Response returned")
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@router.get("/{identifier}/image")
async def get_explanation_image(identifier: int, current_user=Depends(get_current_user)):
    node_id = loader.resolve_node_id(identifier)
    if node_id < 0:
        raise HTTPException(status_code=404, detail="Node not found")
    img_path = STATIC_DIR / f"explain_{node_id}.png"
    if not img_path.exists():
        # Generate it
        await get_explanation(identifier, current_user)
    if img_path.exists():
        return FileResponse(str(img_path), media_type="image/png")
    raise HTTPException(status_code=404, detail="Image not generated")
