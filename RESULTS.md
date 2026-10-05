# Historical retrieval-only benchmark

These are the local measurements recorded on 2026-09-28, not results rerun during packaging. The task checkpoints and locally saved HE rankings were used; the rankings are not included in this package. VPR descriptors and FAISS indexes were already prepared. Timing is the median **single-query search latency** including height-subindex searches and merge, but excluding image decoding, VPR/HE inference, and index construction. CPU `IndexFlatL2` used 8 threads and returned up to 100 candidates. Results depend on hardware and feature/query ordering; reproduce before citing externally.

| Dataset | Mode | Mean candidates | Median search | P95 search | R@1 | R@5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GStudio | Full | 29,768 | 48.232 ms | 52.091 ms | 69.17% | 87.08% |
| GStudio | HE Top-1 | 4,199 | 6.048 ms | 7.251 ms | 65.83% | 85.67% |
| GStudio | HE Top-5 | 10,056 | 18.065 ms | 25.408 ms | 69.25% | 87.50% |
| GStudio | HE Top-10 | 14,056 | 24.383 ms | 35.092 ms | 69.67% | 87.67% |
| Jimo | Full | 36,400 | 59.371 ms | 67.063 ms | 86.24% | 86.95% |
| Jimo | HE Top-1 | 2,600 | 4.236 ms | 5.292 ms | 82.18% | 83.96% |
| Jimo | HE Top-5 | 9,675 | 16.871 ms | 22.337 ms | 85.94% | 86.45% |
| Jimo | HE Top-10 | 14,783 | 25.099 ms | 33.732 ms | 85.94% | 86.65% |

HE Top-K refers to the first K saved height predictions, deduplicated before subindex search. A query with no supported predicted height falls back to the full index; 22 GStudio Top-1 queries did so in the historical run. The release CLI supports full-database recall after new VPR feature extraction, but does not reproduce the HE Top-K modes or benchmark latency.

These numbers demonstrate retrieval-stage speedups only. They do not establish an end-to-end HE-VPR speedup. In a separate GStudio measurement, HE inference plus rereading 1,200 query images took 36.606 s, while batched full-database VPR search took 1.226 s. That measurement did not share query preprocessing between tasks.
