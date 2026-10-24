> Model: GPT-6 / Codex; exact model id unavailable. Style: concise technical reference. Generated: 2026-10-17. Edited: Claude Opus 5.5 (claude-opus-5-5), 2026-10-24.

# Stage 19: 18LITE prototype

Stage 19 evaluates finite exit-field elements and source-ensemble coherence. It has an independent Python/C++ implementation and does not import NumPy, SciPy, Python FINUFFT, or matplotlib. It does not change Stage 14, 17, or 18.

## Invocation and input

```powershell
.\.venv\Scripts\python.exe -B -m formula.capsysred scene.yaml -o out\stage19 --stages 19
```

The YAML contains the ordinary capillary geometry and explicit observation grids. The source archive supplies the exact point-source positions; only section headers and the index are read. The archive ray bodies and their checksums are not read or certified by this operation. New mesh nodes are traced using the selected scene and working precision.

```yaml
stage19:
  source_archive: out/C-2-625/rays-modes
  max_modes: 625
  workers: 1
  observables: [point, coherent_cell]
  screen_indices: [0, 1]
  resume: true
  archive_nodes: none
  archive_compression: gzip
  intensity_floor: 1.0e-6
  minimum_accepted_fraction: 0.95
  preview_modes: [4, 16, 32, 64, 128]
  mesh: {}
  kernel:
    backend: contour
```

`mesh` contains the mesh builder's options. `kernel` contains the propagation options. `contour` selects direct contour integration; `native_nufft` selects the separately checked native accelerated evaluator. The selected backend is part of the saved configuration. Neither name loads the Python FINUFFT package.

The C-2 configuration is experimental: its original 20,000-node budget was insufficient in the first physical pilot. It is not a known successful 625-source recipe.

A later bounded pilot on one actual archived source and one outer bore used 1,000,000 nodes and 1179.27 seconds, accepting 108,181 elements covering 71.55% of that entrance. The rest still needs refinement. No complete C-2 coherence maps or 625-source production run are available yet; do not use the accepted subset as though it were the full aperture.

The 2026-10-18 experimental 5% profile reached 91.30% with the same million-node budget on that source and bore, accepting 140,191 elements in 1251.21 seconds. It still falls below the prototype's 95% coverage gate. On sequential 100,000-node pilots, the selected optimizations reduced mesh time from 135.25 to 81.13 seconds while changing coverage from 24.59% to 28.47%; that speed ratio did not carry over to the larger run. These are mesh experiments with different acceptance tolerances, not validated full C-2 fields.

`mesh.transport_backend` defaults to `decimal`. The optional `interval` mode uses the separate `formula._stage19_transport` C++ extension for a guarded interval evaluation of ray variations, with Decimal DRT fallback on ambiguous roots, endpoints, grazing or determinants. Independent MP position/wall/direction checks remain active. Unreflected rays may use their exact affine fixed-z map after those MP checks. Coordinates, OPL, Fresnel factors, element areas and phase gates remain multiprecision. A successful interval path certifies discrete Maslov/sign decisions under the stated floating-point assumptions; it does not certify the GO field or replace caustic uniformization. Per-bore diagnostics count the backends actually used. This optional path is included in the implementation digest.

`mesh.lazy_physics: true` is an optional rejection prefilter; the default is `false`. It may reject an element using the same MP geometry, OPL or area checks before computing Fresnel and DRT. Every surviving element still passes the full acceptance checks selected by the mesh options. Deferred nodes are explicitly `physics_pending` with `valid: null`, never valid field samples. Earlier rejection can change cache insertion order and the mesh reached under a finite node budget; this option is not a change in the accepted field model. Statistics report computed and pending node counts.

`mesh.check_individual_fresnel` defaults to `true`. Setting it to `false` makes variation of each reflection coefficient diagnostic only; the full complex product is still checked at every existing probe. Accepted elements retain `fresnel_diagnostics.individual_exceedances` and `individual_check_enforced`. Histories, Maslov, geometry, phase, amplitude and density checks remain active.

Acceptance also compares the constant element amplitude and its input-coordinate integration density with local GO values at all seven existing probes. With `j = abs(det Q)`, the probe amplitude is `F / source_distance * sqrt(uz_in / (uz_out * j)) * exp(-i*pi*maslov/2)`, and its density is `A * j`. Both use the existing `eps_A`/`eps_rho` and source-scaled absolute floors. Parent/child finite-area checks remain active. This gate was added after the 2026-10-18 point-J audit; historical meshes do not retroactively pass it.

The interval adapter preserves determinant bounds. The field gate evaluates a conservative enclosure across those bounds; an ambiguous comparison requests Decimal DRT on a repeated trace of the same coordinate. It verifies the cached geometry, OPL, Fresnel, Maslov, orientation and determinant interval before using the result. This exceptional repeat adds no new spatial node and is counted by `determinant_retraces` and `transport_backends.decimal_field_fallback`. Ordinary transport still reuses its original cached trace. Accepted records include `point_field_diagnostics`, which describes the checked probes, not an error bound for the whole element. The amplitude-range arithmetic is Decimal arithmetic, not a fully outward-rounded interval field solver.

The experimental 5% profile, authorized on 2026-10-18: relative amplitude, density and Fresnel tolerances are 5%, the separate parent/child Jacobian tolerance is 10%, and lazy physics plus guarded interval transport are enabled. The phase tolerance stays 0.05 rad and geometry tolerance stays 2%. Local 5% tolerances are not a 5% coherence-error guarantee. The old configuration and builder defaults remain available for comparison.

`mesh.phase_coordinates` defaults to `pullback`, retaining the earlier phase check at the affine image of each input probe. The explicit `physical_exit` mode instead tests the same linear field model at the **actual traced exit point**. It also requires a simple six-vertex output polygon and consistently oriented center fan at the existing probes. The field model remains constant complex amplitude and central linear OPL; the change aligns the phase check with its physical output coordinates.

Physical-exit mode additionally gates each bore on `domain_indicator`: shared input edges of compatible oriented branches use the difference between their complete complex fields, while aperture edges, missing neighbors and incompatible branches use individual amplitude envelopes. Unequal Fresnel/Maslov amplitudes and relative phase anchors are retained. The midpoint-ear indicator uses safety factor 2 and is normalized by the sum of `abs(A)` times straight-element output area; `mesh.domain_relative` defaults to `0.02`. Passing this finite-probe geometric indicator is not a certified curved-boundary, field or coherence error bound. A missing, failed, nonfinite or geometrically invalid diagnostic prevents field computation.

`mesh.target_accepted_fraction` optionally stops refinement once the accepted physical entrance area reaches the requested fraction. The default `null` preserves the budget-driven run. A configured target must be at least `minimum_accepted_fraction` and at most 1. Early stopping does not waive any local field or global domain check; unprocessed faces remain explicitly unresolved. The minimum is a publication gate and the target is a stopping rule, not an error tolerance.

The current C-2 production candidate: 625 recorded sources, physical-exit checks, target/minimum coverage 95%, 2,000,000-node and triangle ceilings per bore, depth 32, FIFO refinement, two source workers and four native threads per worker. Preliminary prefixes are 2, 4, 16, 32, 64, 128 and 256 sources. Native quadrature batches are capped at 2,000,000 source points. It uses precision 32 after comparison with precision 64 on 300 fixed accepted probes and 124 difficult actual rays: no discrete mismatch, maximum phase discrepancy about `4e-22 rad`, and regular-probe amplitude discrepancy about `1.8e-15` for `abs(J) > 1e-16`. The precision-64 configurations remain independent references; requesting 48 digits in this native tracer selects the same precision tier as 64. These checks support the selected numeric precision, not full-ensemble accuracy or a guaranteed disk/runtime budget.

`archive_nodes` controls optional trace-cache provenance: `none` (default) saves no node records; `accepted` saves the node IDs explicitly referenced by accepted elements; `all` saves all cached nodes, including probes of rejected parent triangles. Every policy preserves **all accepted field elements for every source realization**, with unchanged multiprecision input/output coordinates, central OPL, phase/amplitude data and diagnostics, plus all unresolved input elements. This option changes storage, not the field model, mesh acceptance or source ensemble. It does not guarantee that the complete run will fit on a given disk.

`archive_compression` selects the lossless mesh container: `gzip` (default, level 3, `.jsonl.gz`) or `xz` (standard-library LZMA, preset 6, `.jsonl.xz`). Both preserve exactly the same uncompressed JSONL header and MP record text. XZ trades compression time and encoder memory for smaller files; measure available disk space for the intended source count. Coherence-map JSONL exports always remain gzip. Changing this option changes the run contract: existing runs are not automatically converted or resumed under another compression setting.

The C-2 target grids are the actual exit `z = 0.23 m`, `801 × 801` cells across `240 µm`, and the `+35 cm` screen `z = 0.58 m`, `1067 × 1067` cells across `320 µm`. Their cell widths are respectively `240/801 µm` and `320/1067 µm`; neither width is rounded to exactly `0.3 µm`. Each grid stays fixed for every source realization. `n_rays` from the legacy source configuration does not set the adaptive mesh size.

The prototype supports monochromatic circular capillaries and square observation cells. Its reference is the configured cell containing the requested reference position. A reference outside the observation window is rejected. At the exit, the finite field is evaluated on that plane; a zero propagation distance is never passed to the Fresnel propagator.

## Observables

For each realization, `point` evaluates the complex field at each cell center. `coherent_cell` evaluates the complex field averaged over the complete square cell, including zero extension outside the represented exit support. It is not the average intensity inside the cell.

With the same source set in every map, the accumulator records:

```text
W(P) = mean[E_s(P) conjugate(E_s(reference))]
I(P) = mean[|E_s(P)|²]
mu(P) = W(P) / sqrt[I(P) I(reference)]
```

There is no independent stochastic ray self-pair subtraction: each source supplies one deterministic reconstructed field. Invalid or sufficiently dark pixels are masked, not assigned zero coherence. Values above one are diagnosed rather than silently used as evidence of accuracy. The common propagation carrier `exp(i k d)` is omitted within each screen; cross-plane field correlations are not computed.

Full OPL values and geometry remain decimal strings in the source mesh archive. Before sending a field element to the double-precision kernel, its central optical phase is reduced using the full multiprecision wave number and OPL. A large OPL is never converted to double before this phase reduction. Kernel coordinates, slopes and amplitudes are then double precision; their suitability still requires the declared numerical checks.

## Resume and files

The stage directory contains:

- `manifest.json`: geometry, sources, selected grids, options, source-index digest and implementation digests.
- `modes/mode-NNNNNN/mesh.jsonl.gz` (or `.jsonl.xz`): source, every accepted element, diagnostics, unresolved regions and the node records selected by `archive_nodes`. Every realization is retained. The header states the policy and counts of traced, available, saved and omitted nodes.
- `modes/mode-NNNNNN/result.json`: per-source timings, mesh diagnostics and field-buffer checksums.
- `modes/mode-NNNNNN/incomplete.json`: an explicit mesh-acceptance failure, if encountered; no field has been computed for that failed source. It records the failed, saved and missing bores. The first-source diagnostic SVG is explicitly marked partial when construction stops at a failed bore.
- `modes/mode-000000/input-triangles.svg`: the first realization's input partition, with red fill for unresolved elements and the same width and height as its observation map.
- `checkpoint.json` and accumulator binary files: completed source prefix and compensated coherence sums. A checkpoint is published only after all maps for that source are accumulated.
- `<screen>-<observable>/coherence.jsonl.gz`: complex correlation, intensity, complex coherence, magnitude and validity for every cell.
- `<screen>-<observable>/coherence.png`, `coherence.svg`, `intensity.svg`, `meta.json`: maps and their observation contract.
- `meta.json`: completed source count, provenance, timing and limitations.

Per-source field buffers are temporary: after the accumulator checkpoint is committed, they are removed. Mesh archives remain. Resume rejects a different source set, configuration or implementation digest. Source accumulation order is deterministic even when workers finish out of order. The number of simultaneously staged sources is bounded by the worker count.

Optional `preview_modes` exports maps from the completed source prefix under `previews/modes-NNNNNN/`, using its actual count for normalization and explicitly recording the final target count. This does not change the configured ensemble or final maps. A failed preview export preserves the accumulator checkpoint; resume can rebuild that preview. The list must contain distinct positive counts no greater than the target.

During construction, atomic `mesh-bore-NNNN.jsonl.gz` files (or `.jsonl.xz` with XZ selected) preserve completed bores. Resume reuses them only with the identical source, options, screens and implementation digest. Each saved bore must pass its gates before the next bore is traced. A failed bore remains on disk and stops the source; no field from the other bores is published alone. These checkpoints preserve work across interruptions; they do not continue an incomplete bore at a larger budget or reuse a mesh after changing its physical checks.

The driver packs a passed bore's native field rows and immediately releases its MP model and node dictionaries before building the next bore. It assembles the final `mesh.jsonl.gz` or `mesh.jsonl.xz` by streaming the original per-bore rows, verifies declared record counts, and only then removes its known intermediate archives. Resume streams accepted MP records into native rows without loading all MP dictionaries. The first-source SVG uses separate streamed passes for accepted and unresolved elements, preserving drawing order. All MP data already selected by the node policy survive unchanged; the shared archive reader selects gzip or XZ from the filename. Native field rows remain in RAM for the source, so this bounds MP accumulation by one bore rather than promising constant total memory. Combined statistics report `retained_nodes: 0` and the largest completed bore's `peak_bore_retained_nodes`; archive node counts describe persisted provenance independently.

Element node IDs are identifiers in the original per-source, per-bore trace cache, not archive row numbers. In particular, with `archive_nodes: none` they do not point to saved node records. `accepted` retains the explicit center/probe references, but does not claim a complete trace of every subdivision check: a check whose additional node IDs were not recorded in the element is absent. Even `all` stores the node records provided by the mesh builder, not a mandatory archive of all full ray trajectories. The accepted field elements themselves are self-contained for the implemented propagation kernels. The driver releases cached nodes before field evaluation and skips loading them when resuming from a mesh archive; the `none` policy also avoids accumulating completed bores' node records in memory.

Optional `mesh.record_cache_directory` stores cold trace payloads in a private temporary file on the selected volume. `record_cache_capacity` (default 4096, a positive integer) bounds the number of decoded records kept in its LRU. Every cache key and cumulative node ID remains available. Accepted models retain the same mutable record wrappers, so later determinant refinement still updates every alias. Decimal values are serialized exactly; mutable Fresnel coefficient lists retain their identity. Pending native incidence values use the backend's full `str(0)` representation, including guard digits, with a native equality and canonical representation check on every write. Display-precision `Number.parts` is unsuitable for this storage.

This option changes storage, not the physical checks, trace selection or accepted field model. Cache I/O and codec failures abort the run instead of rejecting a physical ray. Files close and disappear after the bore or on exceptions; they are not durable checkpoints for an unfinished bore. Completed bore archives remain the resume boundary. Coordinate keys, wrappers, the adaptive partition and accepted geometry still occupy RAM, so this bounds decoded trace storage rather than total mesh memory. The implementation digest includes the enabled storage helper.

## Interpretation

The output status is experimental. Completion of 625 sources establishes completion of that finite ensemble, not convergence of the mesh, the propagation quadrature, or the source integral. Unresolved input area must be reported alongside the maps. Its omission is not a physical zero and is not repaired by normalization of the coherence.

The 2026-10-18 fresh-ray audit found that parent/child finite-area amplitude and density comparisons do not bound pointwise amplitude variation. A triangle passing the earlier experimental 5% profile differed from direct local GO amplitude by 24.09% at an additional interior probe and 50.69% at an existing vertex. Decimal DRT and independent ray finite differences confirmed the Jacobian; this was not an interval-backend error. The pointwise gate now rejects that element as `point_amplitude`, preserving the central field model and the chosen tolerances. These are local discrepancies and a local acceptance fix, not full-field or coherence-error estimates.

On the same fixed patch, a separate shape-preserving uniform refinement experiment reached full coverage with 4096 elements passing every production field check. Its summed contribution differed from the detailed regular-GO reference by 0.00280027% in relative L2 over nine receivers, versus 12.7217% for the original element. All 12,288 fresh interior probes stayed within 5% amplitude/density tolerances. This validates the local correction under the same propagation model; uniform refinement was a diagnostic, not a replacement production mesher.

The subsequent longest-edge closure fix avoids forcing a neighbor to split its short edge. It preserves conformity and rolls back a propagation chain if a depth or triangle budget prevents completion. The production adaptive partition now covers that same patch completely with 321 elements and 1952 cached nodes in 2.84 seconds. Its nine-receiver relative L2 discrepancy is 0.113064%; all 963 fresh probes passed the 5% amplitude/density checks. This fixes the observed local shape churn, while whole-aperture runtime and caustic accuracy remain separate questions.

The prototype stops if any source has no accepted elements or any bore has less than `minimum_accepted_fraction` of its physical input area accepted. The default `0.95` is an explicit prototype coverage gate, not a previously approved plan tolerance and not a 5% field-error bound. The failed mesh and its diagnostics remain on disk; the source is not skipped or assigned zero weight. A zero ensemble reference intensity also prevents publication of a normalized map.

An independent branch-flux acceptance test is not made mandatory by this driver. Its approval and detailed design remain open. Comparison with older Stage 16 maps must match source ensemble, receiver size, reference and observable; Gauss–Hermite source quadrature is not the same realization set as the recorded Monte Carlo source.
