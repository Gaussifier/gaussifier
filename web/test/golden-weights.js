// Compare provenance before scoring tensors produced by different programs.
export function assertGoldenWeights(bundle, golden) {
  if (golden.model_version !== bundle.model_version ||
      golden.weights_sha256?.stage1 !== bundle.weights_sha256.stage1 ||
      golden.weights_sha256?.head !== bundle.weights_sha256.head) {
    throw new Error("goldens and web bundle use different weights; regenerate both from the same weights directory");
  }
}
