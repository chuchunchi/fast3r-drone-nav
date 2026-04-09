# Feature Specification: Pairwise Cached Inference for Fast3R Homing

## 1. Motivation

During the HOMING phase, the system repeatedly calls `Fast3REngine.infer_pair(live_frame, target_keyframe.image)` inside `HomingController.process_homing_frame()`. Each call runs the full Fast3R forward pass: encoder, decoder, and DPT head on **both** images.

The target keyframe image does not change until the drone reaches the current waypoint and advances to the next keyframe. Across potentially dozens of consecutive frames targeting the same keyframe, the encoder re-processes the identical target image every time.

The encoder (`CroCoEncoder`) is a stack of ViT self-attention blocks (24 layers for ViT-Large, the deployed model). For a 512x384 input, this produces a `(1, P, D)` feature tensor (where P = number of patches, D = embed_dim). Encoding two images means the encoder processes `[2, C, H, W]` through all blocks. By caching the target's encoder output, we skip half the encoder computation on every homing frame.

### Why This Is a Real Model-Side Contribution

This is not a generic software cache. The optimization exploits a structural property of the Fast3R architecture:

- The encoder processes each image **independently** (no cross-view attention in the encoder).
- Cross-view reasoning happens only in the **decoder**, where features from both images are concatenated and attend to each other.
- Therefore, the target image's encoder output is a valid, reusable intermediate representation that can be paired with any new live frame at the decoder stage.

This makes a clean thesis argument: *FAST3R's encoder-decoder separation enables asymmetric inference for sequential pairwise tasks, where one view is stable and the other changes frame-to-frame.*

## 2. Scope

### In Scope

- A new `encode_image()` method on `Fast3R` for encoding a single image.
- A new `forward_pair_cached()` method on `Fast3R` that accepts pre-encoded features for one view and a raw image for the other.
- Updated `Fast3REngine` with `encode_target()` and `infer_pair_cached()` methods.
- Cache management in `HomingController` (store on keyframe set, invalidate on advance).
- Profiling harness to measure encoder time savings.

### Out of Scope

- Changing trained weights or retraining.
- Modifying the decoder or head architecture.
- Multi-view (N>2) caching strategies.
- Caching decoder or head outputs (these depend on both views jointly).

## 3. Architecture Overview

### 3.1 Current Inference Path (per homing frame)

```
live_frame ──┐
             ├─► _encode_images([view0, view1]) ──► decoder ──► head ──► results
target_kf ───┘
             ▲
             │
         Encoder runs on BOTH images every frame.
         Target encoding is redundant while target_idx is unchanged.
```

### 3.2 Proposed Cached Inference Path

```
[On keyframe change]
target_kf ──► encoder ──► cached_target_feats (stored on GPU)

[Per homing frame]
live_frame ──► encoder ──► live_feats
                              │
live_feats + cached_target_feats ──► decoder ──► head ──► results
```

### 3.3 Computational Savings

For a single forward pass with 512x384 inputs on the CroCoEncoder:

| Stage | Uncached (2 images) | Cached (1 image) | Savings |
|-------|--------------------|--------------------|---------|
| Encoder | 2x through all ViT blocks (24 for ViT-Large) | 1x through all ViT blocks | ~50% encoder time |
| Decoder | 1x on concatenated features | 1x on concatenated features | 0% (unchanged) |
| Head | 1x on decoder output | 1x on decoder output | 0% (unchanged) |

The encoder is typically 30-40% of total forward time (based on the existing profiling instrumentation in `fast3r.py`), so the expected end-to-end savings is roughly **15-20% of total inference latency**.

## 4. Detailed Design

### 4.1 Changes to `fast3r/models/fast3r.py` — `Fast3R` Class

#### 4.1.1 New Method: `encode_image()`

Encodes a single prepared view dict and returns the encoder output tuple. This is a thin wrapper around the existing `_encode_images()` for a single-element list.

**Signature:**

```python
def encode_image(self, view: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Encode a single image through the encoder.

    Args:
        view: A single view dict with 'img' (1, C, H, W) and 'true_shape' (1, 2).

    Returns:
        Tuple of (encoded_feat, position, shape):
            encoded_feat: (1, P, D) encoder features
            position: (1, P, 2) positional encoding
            shape: (1, 2) true shape tensor
    """
```

**Implementation approach:**

Call `self.encoder(view["img"], true_shape)` directly, matching the same-shape path in `_encode_images()` but for a single image. Return the three tensors as a tuple.

#### 4.1.2 New Method: `forward_pair_cached()`

Runs the full decoder + head path using one pre-encoded view and one raw view.

**Signature:**

```python
def forward_pair_cached(
    self,
    live_view: dict,
    cached_target: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    profiling: bool = False,
) -> list[dict]:
    """
    Pairwise inference with a pre-encoded target image.

    The live image is encoded on the fly. The cached target's encoder
    output is reused directly. The decoder and head run on the combined
    features exactly as in the standard forward() path.

    Args:
        live_view: View dict for the live frame ('img', 'true_shape').
        cached_target: Tuple of (encoded_feat, position, shape) from
                       a prior encode_image() call for the target keyframe.
        profiling: Whether to return profiling info.

    Returns:
        list[dict]: Two-element list of per-view results, identical in
                    format to forward([live_view, target_view]).
    """
```

**Implementation approach — step by step:**

1. **Encode the live image only:**

   ```python
   live_feat, live_pos = self.encoder(live_view["img"], live_view_true_shape)
   ```

2. **Assemble the same data structures that `forward()` builds:**

   ```python
   encoded_feats = [live_feat, cached_target[0]]  # list of 2 tensors
   positions = [live_pos, cached_target[1]]
   shapes = [live_view_true_shape, cached_target[2]]
   ```

   This is the exact same `(encoded_feats, positions, shapes)` tuple that `_encode_images()` returns. From this point onward, the code path is identical to the existing `forward()`.

3. **Build image IDs:**

   Identical to lines 339-348 of the current `forward()`, but since N=2 and both views have the same resolution (guaranteed by our usage), this simplifies to:

   ```python
   P = encoded_feats[0].shape[1]
   image_ids = torch.cat([
       torch.zeros(P, dtype=torch.long),
       torch.ones(P, dtype=torch.long),
   ]).unsqueeze(0).to(device)
   ```

4. **Run decoder:** `self.decoder(encoded_feats, positions, image_ids)` — unchanged.

5. **Run head and remap:** Same logic as the `else` branch at line 429 of the current `forward()` (same resolution, inference mode). Since this method is inference-only with B=1 and N=2 at the same resolution, only the `else` branch is needed (not the `different_resolution_across_views or self.training` branch).

**Local head handling:** The deployed checkpoint loaded by `Fast3REngine` (`jedyang97/Fast3R_ViT_Large_512`) uses the base config which does not set `with_local_head` (defaults to `False`). The homing path only consumes `pts3d_in_other_view` and `conf` (global head outputs). Therefore, `forward_pair_cached()` does **not** need to process `self.local_head`. If a model with `local_head` is used in the future, this method would need to be extended. Add a guard:

```python
if self.local_head is not None:
    raise NotImplementedError(
        "forward_pair_cached() does not yet support local_head. "
        "Use forward() instead."
    )
```

#### 4.1.3 Why Not Just Refactor `forward()`?

The standard `forward()` must remain unchanged for:
- Training (arbitrary N, mixed resolutions, gradient flow through encoder).
- Evaluation scripts that pass N>2 views.
- Compatibility with checkpoints and the upstream Fast3R codebase.

`forward_pair_cached()` is a separate inference-only method. It shares the same decoder/head code path but has a distinct encoder entry point.

### 4.2 Changes to `visual_homing/server/fast3r_engine.py` — `Fast3REngine` Class

#### 4.2.1 New Method: `encode_target()`

```python
@torch.no_grad()
def encode_target(self, image: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pre-encode a target image for cached pairwise inference.

    Args:
        image: RGB image as numpy array (H, W, 3), uint8.

    Returns:
        Tuple of (encoded_feat, position, shape) that can be passed
        to infer_pair_cached().
    """
```

Prepares the image (resize, normalize, to tensor) and calls `self.model.encode_image(view)`.

**Critical: must wrap in `torch.autocast`** to match the precision context used by the standard `infer_pair()` path. The current inference path goes through `loss_of_one_batch()` which wraps `model(views)` in `torch.autocast(device_type=..., dtype=...)`. The encoder features must be computed under the same autocast context so that:

1. The cached features have the correct dtype (e.g. float16 under autocast, not float32).
2. The features are numerically consistent with what the standard path would produce.

Implementation:

```python
@torch.no_grad()
def encode_target(self, image: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    view = self._prepare_image(image)
    with torch.autocast(device_type=self.device.type, dtype=self._autocast_dtype):
        return self.model.encode_image(view)
```

Where `self._autocast_dtype` is derived from `self.dtype` following the same logic as `loss_of_one_batch()` (see Section 4.2.4 below).

#### 4.2.2 New Method: `infer_pair_cached()`

```python
@torch.no_grad()
def infer_pair_cached(
    self,
    live_image: np.ndarray,
    cached_target: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Run pairwise inference using a pre-encoded target.

    Args:
        live_image: Current camera frame (H, W, 3), uint8.
        cached_target: Output of encode_target() for the reference keyframe.

    Returns:
        Same format as infer_pair(): dict with pts3d_1, pts3d_2, conf_1, conf_2.
    """
```

**Critical: must also wrap in `torch.autocast`** with the same dtype used by `encode_target()` and the standard path:

```python
@torch.no_grad()
def infer_pair_cached(self, live_image, cached_target):
    live_view = self._prepare_image(live_image)
    with torch.autocast(device_type=self.device.type, dtype=self._autocast_dtype):
        preds = self.model.forward_pair_cached(live_view, cached_target)
    # ... unpack results identically to infer_pair() ...
```

Prepares the live image, calls `self.model.forward_pair_cached(live_view, cached_target)`, and unpacks results identically to the existing `infer_pair()`.

#### 4.2.4 Autocast Dtype Resolution

The standard inference path passes `self.dtype` (a `torch.dtype`) as the `precision` argument to `loss_of_one_batch()`, which resolves the autocast dtype as follows:

| `self.dtype` value | `loss_of_one_batch` branch hit | Effective autocast dtype |
|---------------------|-------------------------------|--------------------------|
| `torch.bfloat16`   | `precision == torch.bfloat16` | `torch.bfloat16` |
| `torch.float16`    | Falls through all branches    | Default (`torch.float16`) |
| `torch.float32`    | Falls through all branches    | Default (`torch.float16`) |

**Note:** When `self.dtype` is `torch.float32`, no branch matches (the code checks for string `"32"`, not `torch.float32`), so `torch.autocast(device_type="cuda")` runs with its default: **enabled=True, dtype=torch.float16**. This means the model runs under float16 autocast even when the config says `dtype: "float32"`.

Add a helper property to `Fast3REngine` to encapsulate this logic:

```python
@property
def _autocast_dtype(self) -> torch.dtype:
    """Resolve the autocast dtype to match loss_of_one_batch() behavior."""
    if self.dtype == torch.bfloat16:
        return torch.bfloat16
    return torch.float16  # Default CUDA autocast dtype
```

This ensures `encode_target()` and `infer_pair_cached()` use exactly the same precision as the standard path.

#### 4.2.3 Existing Methods

`infer_pair()` and `infer_multiview()` remain **completely unchanged**. The cached path is an opt-in alternative, not a replacement.

### 4.3 Changes to `visual_homing/server/homing_controller.py` — `HomingController`

#### 4.3.1 New State: Cached Target Encoding

Add to `__init__`:

```python
self._cached_target_encoding = None  # Tuple or None
self._cached_target_idx: int = -1    # Which keyframe index is cached
```

#### 4.3.2 Cache Management

**Set cache** — when the target keyframe changes (in `start_homing()` and `_advance_to_next_keyframe()`):

```python
def _cache_target_encoding(self) -> None:
    """Encode and cache the current target keyframe."""
    if self.target_idx < 0:
        self._cached_target_encoding = None
        self._cached_target_idx = -1
        return

    target_kf = self.keyframe_manager[self.target_idx]
    self._cached_target_encoding = self.fast3r.encode_target(target_kf.image)
    self._cached_target_idx = self.target_idx
```

**Invalidate cache** — on `reset()`, `emergency_stop()`, or if `target_idx` has changed unexpectedly.

#### 4.3.3 Updated `process_homing_frame()` Inference Call

Replace:

```python
result = self.fast3r.infer_pair(live_frame, target_keyframe.image)
```

With:

```python
if (self._cached_target_encoding is not None
        and self._cached_target_idx == self.target_idx):
    result = self.fast3r.infer_pair_cached(live_frame, self._cached_target_encoding)
else:
    self._cache_target_encoding()
    result = self.fast3r.infer_pair_cached(live_frame, self._cached_target_encoding)
```

The fallback to uncached `infer_pair()` can also be kept as a safety net for edge cases, but should not be needed in normal operation.

#### 4.3.4 Integration Points for Cache Invalidation

| Event | Action |
|-------|--------|
| `start_homing()` | Encode and cache target at `target_idx` |
| `_advance_to_next_keyframe()` | Encode and cache new target at `target_idx - 1` |
| `reset()` | Set `_cached_target_encoding = None` |
| `emergency_stop()` | Set `_cached_target_encoding = None` |

### 4.4 Memory Impact

One cached encoder output occupies:

- `encoded_feat`: `(1, P, D)` — for 512x384 with patch_size=16: P = (512/16) * (384/16) = 32 * 24 = 768 patches. With D = 1024 (ViT-Large), this is `1 * 768 * 1024 * 4 bytes = 3.1 MB` in float32 (1.6 MB in float16).
- `position`: `(1, 768, 2)` in int64 = `12 KB`.
- `shape`: `(1, 2)` — negligible.

**Total: ~3 MB in float32, ~1.6 MB in float16.** This is negligible relative to the model's ~1.2 GB of weights and the GPU memory budget of an RTX 4090.

## 5. Correctness Guarantee

The cached path must produce **bit-for-bit identical output** to the uncached path for the same input pair. This is guaranteed by construction:

1. The encoder has no cross-view interaction. `encoder(img_A)` produces the same output whether `img_A` is encoded alone or batched with `img_B`. The only difference is that `_encode_images()` concatenates images along batch dim 0 before encoding. For same-shape images, `encoder([A; B])` produces the same per-image output as `encoder(A)` followed by `encoder(B)`, because:
   - Patch embedding is per-image (no cross-batch interaction).
   - Each ViT block uses self-attention within each sample (no cross-batch attention).
   - Layer norm statistics are per-sample.

2. The decoder receives the same `(encoded_feats, positions, image_ids)` regardless of whether the features came from a batched encoder call or separate calls.

3. The head path is unchanged.

**Caveat on floating-point non-determinism:** When the encoder processes `[A; B]` as a batch of 2, versus `A` alone as a batch of 1, CUDA kernel dispatch and internal batching may produce slightly different floating-point results (different reduction orders in layer norm, different memory layouts for attention). In practice this difference is at the level of float32 epsilon (~1e-7) and has no effect on pose estimation. If exact bit-for-bit equivalence is required for validation, run both images through `encode_image()` individually and compare against the batched path.

### 5.1 Impact of `random_image_idx_embedding` on Equivalence

The deployed Fast3R model (and all training/eval configs in this repo) uses `random_image_idx_embedding: true` in the decoder. This means the decoder generates **fresh random view-ordering positional embeddings** on every forward pass via `_get_random_image_pos()` (in `Fast3RDecoder`) or `_get_random_freqs_cis()` (in `LlamaDecoder`). Concretely:

- View 0 always receives `image_idx_emb[0]` (deterministic).
- View 1 receives `image_idx_emb[random_id]` where `random_id` is drawn from `[1, 999]` via `torch.randperm`.
- The random seed is derived from `torch.randint()` on the global RNG, so two consecutive forward calls with identical inputs produce **different outputs**.

**Why this does NOT break encoder caching:**

The random positional embedding is applied inside the **decoder** (`Fast3RDecoder.forward()`, line 785 of `fast3r.py`), after the `decoder_embed` projection. The encoder has no knowledge of or dependency on `image_idx_emb`. Therefore:

1. The encoder output for the target image is deterministic and reusable.
2. The decoder runs fresh on every call (both cached and uncached paths), generating its own random embeddings each time.
3. The cached path and uncached path exhibit the **same level of stochasticity** — neither is more or less random than the other.

**Impact on equivalence testing:**

Standard `torch.allclose` comparisons between cached and uncached outputs will **fail** — but they would also fail between two uncached calls. The equivalence tests (Sections 8.1–8.3) must account for this by either:

1. **Fixing the RNG seed** before each forward call (`torch.manual_seed(seed)`) so both paths generate the same random embeddings, OR
2. **Temporarily overriding** `model.decoder.random_image_idx_embedding = False` during the test, which forces the decoder to use deterministic sinusoidal embeddings based on `image_ids`, OR
3. **Comparing only encoder outputs** (which are deterministic) and validating end-to-end behavior statistically over many runs.

Option 2 is recommended for unit tests because it isolates the caching logic from decoder stochasticity. Option 1 is a useful secondary validation. Both are included in the updated test plan (Section 8).

## 6. Changes to `_calibrate_scale()`

The scale calibration in `HomingController._calibrate_scale()` runs `self.fast3r.infer_pair()` on consecutive keyframe pairs during the TEACH-to-ARMED transition. This is a one-time batch operation, not part of the real-time loop, so it does **not** benefit from caching (each pair uses two different keyframes). No changes needed here.

## 7. Implementation Plan

### Step 1: Add `encode_image()` to `Fast3R`

- Add the method to `fast3r/models/fast3r.py`.
- It should call `self.encoder(view["img"], true_shape)` and return `(feat, pos, shape)`.
- Keep it simple: assume B=1, single image, inference mode.

### Step 2: Add `forward_pair_cached()` to `Fast3R`

- Add the method to `fast3r/models/fast3r.py`.
- Encode only the live view.
- Assemble `encoded_feats`, `positions`, `shapes` from the live encoding + cached target.
- Build `image_ids`, run the decoder, process head output.
- **Recommended approach:** Inline the relevant head processing code (the `else` branch at line 429, ~55 lines for same-resolution inference without `local_head`) directly into `forward_pair_cached()` rather than refactoring `forward()` into a shared helper. Reasons:
  - `forward()` must remain unchanged for training compatibility and upstream mergeability.
  - `forward_pair_cached()` only needs the B=1, N=2, same-resolution, inference-only path — a small subset of what `forward()` handles.
  - The duplication is minimal (~55 lines of straightforward tensor splitting and remapping) and eliminates the risk of regressing the training path.
  - If `forward()` changes upstream, `forward_pair_cached()` can be updated independently.
- Add a guard: `assert self.local_head is None` (see Section 4.1.2).

### Step 3: Add `encode_target()`, `infer_pair_cached()`, and `_autocast_dtype` to `Fast3REngine`

- Add `_autocast_dtype` property to resolve the effective autocast dtype (see Section 4.2.4).
- `encode_target()`: prepare image, wrap in `torch.autocast(dtype=self._autocast_dtype)`, call `model.encode_image()`, return tuple.
- `infer_pair_cached()`: prepare live image, wrap in same `torch.autocast`, call `model.forward_pair_cached()`, unpack results.
- Mirror the same output format as `infer_pair()`.

### Step 4: Integrate into `HomingController`

- Add cache state fields.
- Add `_cache_target_encoding()` helper.
- Wire cache set/invalidation into `start_homing()`, `_advance_to_next_keyframe()`, `reset()`.
- Switch `process_homing_frame()` to use `infer_pair_cached()`.

### Step 5: Profile and validate

- Compare uncached vs cached latency on real image pairs.
- Validate output equivalence.

## 8. Testing Plan

### 8.1 Unit Test: Encoder Output Equivalence

**Goal:** Verify that encoding an image alone produces the same features as encoding it batched with another image. The encoder is deterministic (no `random_image_idx_embedding` involvement), so this test uses direct comparison.

```python
def test_encode_image_equivalence():
    """
    Encode image A alone via encode_image() and batched with B via
    _encode_images([A, B]). Assert the features for A are equal
    within floating-point tolerance.

    The encoder has no cross-batch interaction (self-attention is
    per-sample, LayerNorm is per-sample), so batched and individual
    encoding should produce identical results up to floating-point
    non-determinism from CUDA kernel dispatch.
    """
    model = load_model()
    model.eval()
    view_a, view_b = make_test_views()

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        # Batched path
        feats_batched, pos_batched, _ = model._encode_images([view_a, view_b])
        feat_a_batched = feats_batched[0]
        pos_a_batched = pos_batched[0]

        # Individual path
        feat_a_solo, pos_a_solo, _ = model.encode_image(view_a)

    # Use relaxed tolerance: float16 epsilon is ~1e-3, accumulated over
    # 12+ ViT blocks. Batched vs. single may differ due to CUDA kernel
    # dispatch and memory layout differences.
    assert torch.allclose(feat_a_batched, feat_a_solo, atol=1e-2, rtol=1e-3)
    assert torch.equal(pos_a_batched, pos_a_solo)
```

### 8.2 Unit Test: Cached Forward Equivalence

**Goal:** Verify that `forward_pair_cached()` produces the same output as `forward()` for the same input pair. Because the decoder uses `random_image_idx_embedding=True` (generating fresh random positional embeddings on every call), two calls to `forward()` with identical inputs will differ. To isolate the caching logic, **temporarily disable the random embedding** during this test.

```python
def test_forward_pair_cached_equivalence():
    """
    Run the same pair through forward() and forward_pair_cached()
    with random_image_idx_embedding disabled for determinism.
    Assert all output tensors match within tolerance.
    """
    model = load_model()
    model.eval()
    view_live, view_target = make_test_views()

    # Disable random image idx embedding to make the decoder deterministic
    original_flag = model.decoder.random_image_idx_embedding
    model.decoder.random_image_idx_embedding = False

    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            # Standard path
            results_standard = model.forward([view_live, view_target])

            # Cached path
            cached = model.encode_image(view_target)
            results_cached = model.forward_pair_cached(view_live, cached)

        for i in range(2):
            for key in results_standard[i]:
                assert torch.allclose(
                    results_standard[i][key],
                    results_cached[i][key],
                    atol=1e-2,
                ), f"Mismatch in view {i}, key '{key}'"
    finally:
        model.decoder.random_image_idx_embedding = original_flag


def test_forward_pair_cached_equivalence_with_fixed_seed():
    """
    Secondary validation: with random_image_idx_embedding=True,
    fix the RNG seed before each call so both paths generate the
    same random positional embeddings.
    """
    model = load_model()
    model.eval()
    view_live, view_target = make_test_views()
    seed = 42

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        # Standard path (seeded)
        torch.manual_seed(seed)
        results_standard = model.forward([view_live, view_target])

        # Cached path (seeded with same seed)
        cached = model.encode_image(view_target)
        torch.manual_seed(seed)
        results_cached = model.forward_pair_cached(view_live, cached)

    for i in range(2):
        for key in results_standard[i]:
            assert torch.allclose(
                results_standard[i][key],
                results_cached[i][key],
                atol=1e-2,
            ), f"Mismatch in view {i}, key '{key}'"
```

### 8.3 Unit Test: Engine-Level Equivalence

**Goal:** Verify that `Fast3REngine.infer_pair_cached()` returns the same pointmaps and confidence as `Fast3REngine.infer_pair()`. Uses the same `random_image_idx_embedding=False` strategy for determinism.

```python
def test_engine_cached_equivalence():
    """
    Compare infer_pair() and infer_pair_cached() at the engine level
    with random image idx embedding disabled for determinism.
    """
    engine = Fast3REngine()
    engine.load_model()
    img1, img2 = load_test_images()

    # Disable random embedding for deterministic comparison
    original_flag = engine.model.decoder.random_image_idx_embedding
    engine.model.decoder.random_image_idx_embedding = False

    try:
        result_standard = engine.infer_pair(img1, img2)
        cached_target = engine.encode_target(img2)
        result_cached = engine.infer_pair_cached(img1, cached_target)

        for key in ["pts3d_1", "pts3d_2", "conf_1", "conf_2"]:
            assert torch.allclose(
                result_standard[key], result_cached[key], atol=1e-2
            ), f"Mismatch in '{key}'"
    finally:
        engine.model.decoder.random_image_idx_embedding = original_flag
```

### 8.4 Unit Test: Cache Invalidation

**Goal:** Verify that the HomingController invalidates and refreshes the cache at the correct lifecycle events.

```python
def test_cache_invalidation():
    """
    Simulate a homing run. Verify that:
    1. Cache is populated on start_homing().
    2. Cache is refreshed on _advance_to_next_keyframe().
    3. Cache is cleared on reset().
    """
    controller = make_mock_controller(num_keyframes=3)

    controller.start_homing()
    assert controller._cached_target_idx == 2
    assert controller._cached_target_encoding is not None

    controller._advance_to_next_keyframe()
    assert controller._cached_target_idx == 1

    controller._advance_to_next_keyframe()
    assert controller._cached_target_idx == 0

    controller.reset()
    assert controller._cached_target_encoding is None
    assert controller._cached_target_idx == -1
```

### 8.5 Performance Test: Latency Comparison

**Goal:** Measure wall-clock savings from cached inference.

```python
def test_cached_inference_latency():
    """
    Run N iterations of uncached and cached inference on the same image
    pair. Report mean and std of per-frame latency for each path.
    Use model.forward(profiling=True) and model.forward_pair_cached(profiling=True)
    to get per-stage breakdown.
    """
    model = load_model()
    view_live, view_target = make_test_views()

    # Warmup
    for _ in range(5):
        model.forward([view_live, view_target])

    N = 50

    # Uncached
    times_uncached = []
    for _ in range(N):
        torch.cuda.synchronize()
        t0 = time.time()
        model.forward([view_live, view_target])
        torch.cuda.synchronize()
        times_uncached.append(time.time() - t0)

    # Cached
    cached = model.encode_image(view_target)
    times_cached = []
    for _ in range(N):
        torch.cuda.synchronize()
        t0 = time.time()
        model.forward_pair_cached(view_live, cached)
        torch.cuda.synchronize()
        times_cached.append(time.time() - t0)

    mean_uncached = np.mean(times_uncached) * 1000
    mean_cached = np.mean(times_cached) * 1000
    speedup_pct = (1 - mean_cached / mean_uncached) * 100

    print(f"Uncached: {mean_uncached:.1f} ms")
    print(f"Cached:   {mean_cached:.1f} ms")
    print(f"Speedup:  {speedup_pct:.1f}%")

    assert speedup_pct > 5, "Expected measurable speedup from caching"
```

### 8.6 Integration Test: Homing Sequence with Cache

**Goal:** Run a simulated homing sequence using cached inference and verify that navigation behavior matches the uncached path.

**Note on `random_image_idx_embedding`:** With the random embedding enabled (as in production), the uncached and cached paths will produce *slightly different* commands on each frame (since each forward pass gets new random embeddings regardless of caching). The integration test should verify that both paths produce *statistically similar* navigation behavior, not bit-identical commands. Specifically:

1. Record a short teach sequence of 3-5 keyframes with synthetic images.
2. Run homing with `infer_pair()` (uncached) N times and record the distribution of waypoint-reached events and final positions.
3. Run the same sequence with cached inference N times.
4. Assert that both paths reach the same waypoints in the same order, and that mean navigation error is within tolerance.

For a deterministic variant, disable `random_image_idx_embedding` and assert exact command equivalence (as in Sections 8.2–8.3).

### 8.7 Profiling Breakdown Test

**Goal:** Get a per-stage timing breakdown to confirm where savings come from, suitable for inclusion in a thesis results table.

Use the existing `profiling=True` flag. Report:

| Metric | Uncached | Cached | Delta |
|--------|----------|--------|-------|
| `encode_images_time` | X ms | Y ms | -Z% |
| `decoder_time` | X ms | X ms | 0% |
| `head_forward_time` | X ms | X ms | 0% |
| `total_time` | X ms | Y ms | -Z% |

This table directly supports the thesis claim that the optimization targets the encoder stage without affecting decoder/head behavior.

## 9. Thesis Framing

### Suggested Narrative

> FAST3R was designed as a general-purpose multi-view 3D reconstruction model. Its encoder processes each image independently, while the decoder fuses information across views through joint self-attention. We observe that in sequential pairwise tasks — such as visual homing, where one image (the target keyframe) remains fixed across many inference cycles — this encoder independence enables an asymmetric inference strategy. By pre-computing and caching the encoder output for the stable reference image, we eliminate redundant encoder computation on every frame. This yields a ~15-20% reduction in per-frame inference latency without any modification to the model weights, decoder architecture, or head. The optimization is specific to the pairwise deployment context and demonstrates how a general multi-view model can be efficiently adapted for real-time two-view applications.

### What Makes This a Valid Model-Side Contribution

1. **It exploits a structural property of the architecture** (encoder independence), not just a software-level cache.
2. **It produces measurable, hardware-independent speedup** (fewer FLOPs in the encoder).
3. **It is specific to the thesis application** (sequential pairwise inference with a fixed reference).
4. **It is empirically verifiable** with profiling and equivalence tests.
5. **It requires no retraining**, making it immediately deployable.

### Comparison to the Original Proposal Levels

| Aspect | Original Level 1-3 | Cached Inference |
|--------|---------------------|------------------|
| Where savings come from | Python bookkeeping | Encoder FLOPs |
| Expected speedup | <1% (unmeasurable) | 15-20% (measurable) |
| Requires weight changes | No (L1-2) / Ambiguous (L3) | No |
| Decoder/head changes | Code refactoring | None |
| Thesis argument | "Cleaner code" | "Architectural exploitation" |
| Empirical validation | Hard to show meaningful delta | Clear before/after profiling |

## 10. Future Extensions

These are out of scope for the current implementation but worth noting:

- **Decoder KV caching:** If the decoder used causal attention, the target view's key/value projections could also be cached. FAST3R uses bidirectional attention, so this does not apply without architectural changes that would require retraining.
- **Multi-keyframe lookahead encoding:** During `_advance_to_next_keyframe()`, the next target could be pre-encoded in a background thread while the current frame is still being processed.
- **Batch encoding during TEACH-to-ARMED transition:** All keyframe images could be pre-encoded in one batched pass during `_calibrate_scale()`, so that `start_homing()` has zero-cost cache initialization.
