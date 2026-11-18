// Standalone check / benchmark for the EMV sampler's Voronoi assignment (CMake target
// sampler_assign_check, built with -DGFUSED_BUILD_CHECKS=ON).
//
// Runs gfused::run_sampler on a density map. With --check, every Voronoi assignment (merge
// rounds, Lloyd iterations, final masses) is compared with a brute-force argmin over all seeds
// (same distance arithmetic and lowest-index tie-break); the fused per-cell sums are compared
// with fp64 sums over the brute-force owners, and the merge fallback's nearest-survivor search
// is compared with brute force on random alive masks. Each merge round's neighbour sets are
// compared with the 4-adjacency of the owner map, and its candidate set and matching (cand,
// matched, target, alive, pending) with the former argsort + two-kernels-per-column matching
// on the same masses and neighbour tables. --jfa also runs the former jump-flooding
// + lost-point-recovery assignment on the same seeds and counts its disagreements. --stats
// prints at which ring / level the pixels were resolved. --reps R times the sampler alone under
// CUDA-graph replay (median of R).
//
//   cmake -S cpp_inference/fused -B build-fused -DGFUSED_BUILD_CHECKS=ON
//   cmake --build build-fused --target sampler_assign_check
//   (or: nvcc -std=c++17 -O3 --use_fast_math -arch=sm_89 -o sampler_assign_check
//        cpp_inference/fused/sampler_assign_check.cu)
//   sampler_assign_check --density D.npy (--rate RATE.npy | --n N) [--check] [--jfa] [--stats]
//       [--reps R] [--seed S]
//
// D.npy: float32 [H, W] density (e.g. the test_pipeline_fused.py *.ref.density.npy dumps);
// n = floor(rate) as in the pipeline. Prints one JSON line (counts, mismatch totals, timing).
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace gfused { struct SamplerBuffers; namespace { struct AssignOut; } }
static void check_hook(gfused::SamplerBuffers& b, const void* pts, const int* n_ptr,
                       const int* gate, const gfused::AssignOut& out, cudaStream_t s);
#define GFUSED_SAMPLER_ASSIGN_HOOK(b, pts, n_ptr, gate, out, s) check_hook(b, pts, n_ptr, gate, out, s)
static void match_hook(gfused::SamplerBuffers& b, cudaStream_t s);
#define GFUSED_SAMPLER_MATCH_HOOK(b, s) match_hook(b, s)
#include "pipeline_sampler.cu"

#define CK(x) do { cudaError_t e_ = (x); if (e_) { fprintf(stderr, "CUDA %s line %d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(2);} } while (0)

static std::vector<float> load_npy(const std::string& path, std::vector<long>* shape) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
  char magic[6]; f.read(magic, 6);
  unsigned char ver[2]; f.read((char*)ver, 2);
  uint32_t hlen = 0;
  if (ver[0] == 1) { unsigned char b[2]; f.read((char*)b, 2); hlen = b[0] | (b[1] << 8); }
  else { unsigned char b[4]; f.read((char*)b, 4); hlen = b[0] | (b[1] << 8) | (b[2] << 16) | ((uint32_t)b[3] << 24); }
  std::string header(hlen, '\0'); f.read(&header[0], hlen);
  if (header.find("<f4") == std::string::npos) { fprintf(stderr, "need f4: %s\n", path.c_str()); exit(1); }
  auto lp = header.find('(', header.find("'shape'")); auto rp = header.find(')', lp);
  std::stringstream ss(header.substr(lp + 1, rp - lp - 1)); std::string tok; shape->clear(); size_t total = 1;
  while (std::getline(ss, tok, ',')) { if (tok.find_first_not_of(" ") == std::string::npos) continue; shape->push_back(std::stol(tok)); total *= shape->back(); }
  std::vector<float> v(total); f.read((char*)v.data(), total * 4); return v;
}

// ---------------------------------------------------------------- checking
static bool g_check = false, g_jfa = false, g_stats = false;
static int g_H, g_W;
static int* d_bf = nullptr; static float* d_bfd = nullptr;
static int* d_jo = nullptr; static float* d_jd = nullptr; static float2* d_jp = nullptr;
static int* d_cnt = nullptr; static uint8_t* d_seen = nullptr; static int* d_lost = nullptr;
struct Acc { long long px = 0, wrong = 0, wrong_strict = 0, jwrong = 0, jwrong_strict = 0; int calls = 0; };
static Acc g_acc;

__global__ void h_bf(const float2* __restrict__ pts, int n, int H, int W, int* bo_out, float* bd_out) {
  const int x = blockIdx.x * 16 + threadIdx.x, y = blockIdx.y * 16 + threadIdx.y;
  const bool in = x < W && y < H;
  const float mx = x + 0.5f, my = y + 0.5f;
  float bd = INFINITY; int bo = INT_MAX;
  __shared__ float2 sp[256];
  const int lin = threadIdx.y * 16 + threadIdx.x;
  for (int t0 = 0; t0 < n; t0 += 256) {
    if (t0 + lin < n) { float2 p = pts[t0 + lin];
      sp[lin] = make_float2(fminf(fmaxf(p.x * W, 0.f), W - 1e-3f), fminf(fmaxf(p.y * H, 0.f), H - 1e-3f)); }
    __syncthreads();
    const int m = min(256, n - t0);
    if (in) for (int j = 0; j < m; ++j) {
      const float d = __fadd_rn(__fmul_rn(mx - sp[j].x, mx - sp[j].x), __fmul_rn(my - sp[j].y, my - sp[j].y));
      if (d < bd) { bd = d; bo = t0 + j; }  // ascending j: first minimum = lowest index
    }
    __syncthreads();
  }
  if (in) { bo_out[y * W + x] = bo; bd_out[y * W + x] = bd; }
}
__global__ void h_cmp(const int* own, const int* bf, const float* bfd, const float2* __restrict__ pts, int H, int W, int* cnt) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= H * W) return;
  const int o = own[i];
  if (o == bf[i]) return;
  atomicAdd(&cnt[0], 1);
  float d = INFINITY;
  if (o >= 0) { float2 p = pts[o]; p = make_float2(fminf(fmaxf(p.x * W, 0.f), W - 1e-3f), fminf(fmaxf(p.y * H, 0.f), H - 1e-3f));
    const int x = i % W, y = i / W; const float dx = x + 0.5f - p.x, dy = y + 0.5f - p.y;
    d = __fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy)); }
  if (d != bfd[i]) atomicAdd(&cnt[1], 1);
}

// legacy JFA + recovery (copied from the pre-grid sampler)
__global__ void j_reset(int* owner, float* dist, int P) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < P) { owner[i] = -1; dist[i] = INFINITY; } }
__global__ void j_pix(const float2* pts, int n, float2* pix, int H, int W) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; float2 p = pts[i];
  pix[i] = make_float2(fminf(fmaxf(p.x * W, 0.f), W - 1e-3f), fminf(fmaxf(p.y * H, 0.f), H - 1e-3f)); }
__global__ void j_snap(const float2* pix, int n, int* owner, int H, int W) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; float2 p = pix[i];
  int px = min(max((int)floorf(p.x), 0), W - 1), py = min(max((int)floorf(p.y), 0), H - 1); owner[py * W + px] = i; }
__global__ void j_init(const int* owner, float* dist, const float2* pix, int H, int W) { int x = blockIdx.x * 16 + threadIdx.x, y = blockIdx.y * 16 + threadIdx.y;
  if (x >= W || y >= H) return; int idx = y * W + x, o = owner[idx]; if (o < 0) { dist[idx] = INFINITY; return; }
  float dx = x + 0.5f - pix[o].x, dy = y + 0.5f - pix[o].y; dist[idx] = dx * dx + dy * dy; }
__global__ void j_step(int* owner, float* dist, const float2* pix, int H, int W, int step) {
  const int x = blockIdx.x * 16 + threadIdx.x, y = blockIdx.y * 16 + threadIdx.y; if (x >= W || y >= H) return;
  const int idx = y * W + x; const float mx = x + 0.5f, my = y + 0.5f; int bo = owner[idx]; float bd = dist[idx];
  for (int dyi = -1; dyi <= 1; ++dyi) { int ny = y + dyi * step; if (ny < 0 || ny >= H) continue;
    for (int dxi = -1; dxi <= 1; ++dxi) { if (!dxi && !dyi) continue; int nx = x + dxi * step; if (nx < 0 || nx >= W) continue;
      int c = owner[ny * W + nx]; if (c < 0) continue; float2 q = pix[c]; float dx = mx - q.x, dy = my - q.y; float d = dx * dx + dy * dy;
      if (d < bd) { bd = d; bo = c; } } }
  owner[idx] = bo; dist[idx] = bd; }
__global__ void j_seen(const int* owner, uint8_t* seen, int P) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < P && owner[i] >= 0) seen[owner[i]] = 1; }
__global__ void j_lost(const uint8_t* seen, int n, int* lost, int* cnt) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n && !seen[i]) lost[atomicAdd(cnt, 1)] = i; }
__global__ void j_recover(int* owner, float* dist, const float2* pix, const int* lost, const int* cnt, int H, int W) {
  int x = blockIdx.x * 16 + threadIdx.x, y = blockIdx.y * 16 + threadIdx.y; if (x >= W || y >= H) return; int idx = y * W + x;
  float mx = x + 0.5f, my = y + 0.5f; int bo = owner[idx]; float bd = dist[idx];
  for (int j = 0; j < *cnt; ++j) { float2 q = pix[lost[j]]; float dx = mx - q.x, dy = my - q.y; float d = dx * dx + dy * dy; if (d < bd) { bd = d; bo = lost[j]; } }
  owner[idx] = bo; dist[idx] = bd; }

// nearest-survivor reference: brute force, same metric (rn ops), packed (d, idx) min
__global__ void h_nn_bf(const float2* pts, const uint8_t* alive, int n, const int* list, int count, unsigned long long* best) {
  const int e = blockIdx.y; if (e >= count) return; const float2 p = pts[list[e]]; unsigned long long b = ~0ULL;
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) { if (!alive[j]) continue;
    const float d = __fadd_rn(__fmul_rn(pts[j].x - p.x, pts[j].x - p.x), __fmul_rn(pts[j].y - p.y, pts[j].y - p.y));
    const unsigned long long k = ((unsigned long long)__float_as_uint(d) << 32) | (unsigned)j; b = k < b ? k : b; }
  atomicMin(&best[e], b); }
__global__ void h_nn_setup(uint8_t* alive, int n, int* list, int* count, uint32_t seed, int stride) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  uint32_t h = (i + 1) * 2654435761u ^ seed; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  alive[i] = (h % 3) != 0;
  if (!alive[i] && (h >> 8) % stride == 0) list[atomicAdd(count, 1)] = i; }
static long long g_nn_tests = 0, g_nn_wrong = 0;
static void run_jfa(const float2* pts, int n, int H, int W) {
  const int P = H * W; dim3 jb(16, 16), jg((W + 15) / 16, (H + 15) / 16);
  j_reset<<<(P + 255) / 256, 256>>>(d_jo, d_jd, P);
  j_pix<<<(n + 255) / 256, 256>>>(pts, n, d_jp, H, W);
  j_snap<<<(n + 255) / 256, 256>>>(d_jp, n, d_jo, H, W);
  j_init<<<jg, jb>>>(d_jo, d_jd, d_jp, H, W);
  int cap = std::max(1, std::max(H, W) / 2), step = 1; while (step < cap) step *= 2;
  std::vector<int> sch; while (step >= 1) { sch.push_back(step); step /= 2; } sch.push_back(1);
  for (int st : sch) j_step<<<jg, jb>>>(d_jo, d_jd, d_jp, H, W, st);
  CK(cudaMemset(d_seen, 0, n)); CK(cudaMemset(d_cnt + 2, 0, 4));
  j_seen<<<(P + 255) / 256, 256>>>(d_jo, d_seen, P);
  j_lost<<<(n + 255) / 256, 256>>>(d_seen, n, d_lost, d_cnt + 2);
  j_recover<<<jg, jb>>>(d_jo, d_jd, d_jp, d_lost, d_cnt + 2, H, W);
}

__global__ void h_acc(const int* own, const float* dens, bool lloyd, int H, int W, double* w, double* xa, double* ya) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= H * W) return; const int o = own[i]; const float d = dens[i];
  if (lloyd && d <= 0.f) return; const int x = i % W, y = i / W;
  atomicAdd(&w[o], (double)d); if (lloyd) { atomicAdd(&xa[o], (double)(d * ((x + 0.5f) / W))); atomicAdd(&ya[o], (double)(d * ((y + 0.5f) / H))); } }
__global__ void h_acc_cmp(const float* a, const double* r, int n, double* err) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  const double e = fabs((double)a[i] - r[i]) / (fabs(r[i]) + 1e-3); atomicMax((unsigned long long*)err, __double_as_longlong(e)); }
static double* d_ref3 = nullptr; static double* d_err = nullptr; static double g_acc_err = 0;
static void check_hook(gfused::SamplerBuffers& b, const void* pts_v, const int* n_ptr, const int* gate, const gfused::AssignOut& out, cudaStream_t s) {
  if (!g_check) return;
  const int* own_ptr = out.owner;
  CK(cudaStreamSynchronize(s));
  int n = 0, gv = 1; CK(cudaMemcpy(&n, n_ptr, 4, cudaMemcpyDeviceToHost));
  if (gate) CK(cudaMemcpy(&gv, gate, 4, cudaMemcpyDeviceToHost));
  if (gv <= 0) return;
  const float2* pts = (const float2*)pts_v; const int H = g_H, W = g_W, P = H * W;
  dim3 jb(16, 16), jg((W + 15) / 16, (H + 15) / 16);
  static int* d_own = nullptr; if (!d_own) CK(cudaMalloc(&d_own, P * 4));
  if (!own_ptr) {  // fused accumulation only: re-run the query for the owner map
    CK(cudaMemset(b.grid_open_count, 0, 4)); CK(cudaMemset(b.grid_open_mask, 0, 4 * (((W + 7) / 8) * ((H + 3) / 4))));
    gfused::AssignOut o2{d_own, nullptr, nullptr, nullptr, nullptr};
    gfused::k_grid_query<<<dim3(((W + 1) / 2 + gfused::kQBX - 1) / gfused::kQBX, ((H + 1) / 2 + gfused::kQBY - 1) / gfused::kQBY), dim3(gfused::kQBX, gfused::kQBY)>>>(n_ptr, gate, b.grid_start, b.grid_pos, b.grid_idx, o2, b.grid_open_count, b.grid_open_tiles, b.grid_open_mask, b.grid_open_best, H, W, nullptr);
    gfused::k_grid_query_open<<<256, 128>>>(n_ptr, gate, b.grid_start, b.grid_pos, b.grid_idx, b.grid_open_count, b.grid_open_tiles, b.grid_open_mask, b.grid_open_best, o2, H, W, nullptr);
    own_ptr = d_own;
  }
  h_bf<<<jg, jb>>>(pts, n, H, W, d_bf, d_bfd);
  if (out.dens) {  // fused sums vs double-precision sums over the brute-force owners
    if (!d_ref3) { CK(cudaMalloc(&d_ref3, 3 * 8 * (size_t)b.init_cap)); CK(cudaMalloc(&d_err, 8)); }
    CK(cudaMemset(d_ref3, 0, 3 * 8 * (size_t)b.init_cap)); CK(cudaMemset(d_err, 0, 8));
    const bool lloyd = out.acc_x != nullptr; double* r = d_ref3;
    h_acc<<<(P + 255) / 256, 256>>>(d_bf, out.dens, lloyd, H, W, r, r + b.init_cap, r + 2 * b.init_cap);
    h_acc_cmp<<<(n + 255) / 256, 256>>>(out.acc_w, r, n, d_err);
    if (lloyd) { h_acc_cmp<<<(n + 255) / 256, 256>>>(out.acc_x, r + b.init_cap, n, d_err); h_acc_cmp<<<(n + 255) / 256, 256>>>(out.acc_y, r + 2 * b.init_cap, n, d_err); }
    double e; CK(cudaMemcpy(&e, d_err, 8, cudaMemcpyDeviceToHost)); g_acc_err = std::max(g_acc_err, e);
    fprintf(stderr, "    fused %s sums: max rel err vs fp64 brute-force sums %.3g\n", lloyd ? "lloyd" : "mass", e);
  }
  CK(cudaMemset(d_cnt, 0, 8));
  h_cmp<<<(P + 255) / 256, 256>>>(own_ptr, d_bf, d_bfd, pts, H, W, d_cnt);
  int c[2]; CK(cudaMemcpy(c, d_cnt, 8, cudaMemcpyDeviceToHost));
  g_acc.wrong += c[0]; g_acc.wrong_strict += c[1];
  int jc[2] = {0, 0};
  if (g_jfa) { run_jfa(pts, n, H, W); CK(cudaMemset(d_cnt, 0, 8));
    h_cmp<<<(P + 255) / 256, 256>>>(d_jo, d_bf, d_bfd, pts, H, W, d_cnt);
    CK(cudaMemcpy(jc, d_cnt, 8, cudaMemcpyDeviceToHost)); g_acc.jwrong += jc[0]; g_acc.jwrong_strict += jc[1]; }
  g_acc.px += P; g_acc.calls++;
  if (out.owner) {  // nearest-survivor search on this round's grid vs brute force
    static uint8_t *d_alive = nullptr, *d_matched = nullptr; static int *d_list, *d_tgt; static unsigned long long* d_best; static gfused::SamplerCtrl* d_c;
    if (!d_alive) { CK(cudaMalloc(&d_alive, b.init_cap)); CK(cudaMalloc(&d_matched, b.init_cap)); CK(cudaMalloc(&d_list, 4 * b.init_cap));
      CK(cudaMalloc(&d_tgt, 4 * b.init_cap)); CK(cudaMalloc(&d_best, 8 * b.init_cap)); CK(cudaMalloc(&d_c, sizeof(gfused::SamplerCtrl))); }
    for (int stride : {1, 50}) {
      gfused::SamplerCtrl hc{}; hc.round_remove = 1; hc.round_n_in = n; hc.nn_count = 0;
      CK(cudaMemcpy(d_c, &hc, sizeof(hc), cudaMemcpyHostToDevice));
      h_nn_setup<<<(n + 255) / 256, 256>>>(d_alive, n, d_list, &d_c->nn_count, 777u + stride, stride);
      CK(cudaMemcpy(&hc, d_c, sizeof(hc), cudaMemcpyDeviceToHost)); const int cnt = hc.nn_count;
      CK(cudaMemset(d_best, 0xff, 8 * cnt)); CK(cudaMemset(d_tgt, 0xff, 4 * b.init_cap));
      for (int e0 = 0; e0 < cnt; e0 += 65535) {
        const int m = std::min(65535, cnt - e0);
        h_nn_bf<<<dim3(64, m), 256>>>(pts, d_alive, n, d_list + e0, m, d_best + e0); }
      gfused::k_fallback_nn<<<gfused::kNnBlocks, 128>>>(d_c, pts, d_alive, d_list, b.grid_start, b.grid_idx, d_tgt, d_matched, nullptr, nullptr, nullptr, H, W);
      std::vector<int> list(cnt), tgt(b.init_cap); std::vector<unsigned long long> best(cnt);
      CK(cudaMemcpy(list.data(), d_list, 4 * cnt, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(best.data(), d_best, 8 * cnt, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(tgt.data(), d_tgt, 4 * b.init_cap, cudaMemcpyDeviceToHost));
      int wrong = 0; for (int e = 0; e < cnt; ++e) { const int ref = best[e] == ~0ULL ? -1 : (int)(best[e] & 0xffffffffULL); wrong += tgt[list[e]] != ref; }
      g_nn_tests += cnt; g_nn_wrong += wrong;
      fprintf(stderr, "    nearest-survivor: %d sources, %d mismatches vs brute force\n", cnt, wrong);
    }
  }
  if (g_stats)
  { static int* d_st = nullptr; if (!d_st) CK(cudaMalloc(&d_st, 4 * 64)); CK(cudaMemset(d_st, 0, 256));
    CK(cudaMemset(b.grid_open_count, 0, 4)); CK(cudaMemset(b.grid_open_mask, 0, 4 * (((W + 7) / 8) * ((H + 3) / 4))));
    gfused::AssignOut o3{d_own, nullptr, nullptr, nullptr, nullptr};
    gfused::k_grid_query<<<dim3(((W + 1) / 2 + gfused::kQBX - 1) / gfused::kQBX, ((H + 1) / 2 + gfused::kQBY - 1) / gfused::kQBY), dim3(gfused::kQBX, gfused::kQBY)>>>(n_ptr, gate, b.grid_start, b.grid_pos, b.grid_idx, o3, b.grid_open_count, b.grid_open_tiles, b.grid_open_mask, b.grid_open_best, H, W, d_st);
    gfused::k_grid_query_open<<<256, 128>>>(n_ptr, gate, b.grid_start, b.grid_pos, b.grid_idx, b.grid_open_count, b.grid_open_tiles, b.grid_open_mask, b.grid_open_best, o3, H, W, d_st);
    int st[40]; CK(cudaMemcpy(st, d_st, 160, cudaMemcpyDeviceToHost)); gfused::GridCtrl gh; CK(cudaMemcpy(&gh, b.grid, sizeof(gh), cudaMemcpyDeviceToHost));
    fprintf(stderr, "    s=%d grid %dx%d stats(L=local,0..3 global level:ring=count):", gh.s, gh.gx, gh.gy);
    for (int l = 0; l < 5; ++l) for (int r = 0; r < 8; ++r) if (st[l * 8 + r]) fprintf(stderr, " %c:%d=%d", l ? char('0' + l - 1) : 'L', r, st[l * 8 + r]);
    fprintf(stderr, "\n"); }
  fprintf(stderr, "  assign n=%d: grid mismatch %d (non-tie %d) | jfa mismatch %d (non-tie %d) of %d px\n", n, c[0], c[1], jc[0], jc[1], P);
}


// ---------------------------------------------------------------- matching reference
// The former implementation: stable CUB argsort of the masses -> ranks -> 2 kernels per column.
__global__ void r_keys(const float* mass, int n, int cap, float* k, int* v) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= cap) return; k[i] = i < n ? mass[i] : INFINITY; v[i] = i; }
__global__ void r_init(const int* sv, int n, int R, int* prio, uint8_t* cand, uint8_t* alive, uint8_t* matched, int* target) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x; if (k >= n) return; const int i = sv[k];
  prio[i] = k; cand[i] = k < R; alive[i] = 1; matched[i] = 0; target[i] = -1; }
__global__ void r_p1(const int* prio, const int* table, const uint8_t* cand, const uint8_t* alive, const uint8_t* matched, int n, int K, int col, int* tmin, int* myp) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; const int prop = table[i * K + col];
  const bool valid = cand[i] && alive[i] && !matched[i] && prop != i && prop >= 0 && prop < n && alive[prop];
  const int p = valid ? prio[i] : INT_MAX; myp[i] = p; if (valid) atomicMin(&tmin[prop], p); }
__global__ void r_p2(const int* table, const int* myp, const int* tmin, int* tnext, int* target, uint8_t* matched, uint8_t* alive, int n, int K, int col) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; tnext[i] = INT_MAX; const int p = myp[i];
  if (p == INT_MAX) return; const int prop = table[i * K + col]; if (p == tmin[prop]) { target[i] = prop; matched[i] = 1; alive[i] = 0; } }
__global__ void r_mark(const uint8_t* cand, const uint8_t* matched, uint8_t* alive, uint8_t* pend, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; pend[i] = cand[i] && !matched[i]; if (pend[i]) alive[i] = 0; }
__global__ void r_fill(int* p, int v, int n) { const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) p[i] = v; }
static long long g_match_rounds = 0, g_match_wrong = 0;
static long long g_adj_wrong = 0;
static void match_hook(gfused::SamplerBuffers& b, cudaStream_t s) {
  static int call = 0;
  const int round = call++ % std::max(1, b.cfg.merge_rounds);
  if (!g_check) return;
  CK(cudaStreamSynchronize(s));
  gfused::SamplerCtrl hc; CK(cudaMemcpy(&hc, b.ctrl, sizeof(hc), cudaMemcpyDeviceToHost));
  const int n = hc.round_n_in, R = hc.round_remove, cap = b.init_cap, K = b.cfg.knn_k;
  if (R <= 0) return;
  {  // neighbour sets vs the 4-adjacency of the owner map
    const int H = b.cfg.H, W = b.cfg.W, tag = round & 1;
    std::vector<int> own((size_t)H * W), rows((size_t)n * gfused::kNbrSlots);
    CK(cudaMemcpy(own.data(), b.owner, 4 * own.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(rows.data(), b.nbr, 4 * rows.size(), cudaMemcpyDeviceToHost));
    std::vector<std::vector<int>> ref(n);
    auto add = [&](int a, int c) { if (a >= 0 && c >= 0 && a != c) { ref[a].push_back(c); ref[c].push_back(a); } };
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) { const int o = own[(size_t)y * W + x];
      if (x + 1 < W) add(o, own[(size_t)y * W + x + 1]); if (y + 1 < H) add(o, own[(size_t)(y + 1) * W + x]); }
    long long wrong = 0;
    for (int i = 0; i < n; ++i) {
      std::sort(ref[i].begin(), ref[i].end()); ref[i].erase(std::unique(ref[i].begin(), ref[i].end()), ref[i].end());
      std::vector<int> got;
      for (int q = 0; q < gfused::kNbrSlots; ++q) { const int e = rows[(size_t)i * gfused::kNbrSlots + q]; if (e >= 0 && (e >> 30) == tag) got.push_back(e & 0x3fffffff); }
      std::sort(got.begin(), got.end());
      if (ref[i].size() <= (size_t)gfused::kNbrSlots) wrong += got != ref[i];
      else wrong += got.size() != (size_t)gfused::kNbrSlots;  // overflowing cells keep 16 of them
    }
    g_adj_wrong += wrong;
    fprintf(stderr, "    neighbour sets (round %d): %lld rows differ from the owner-map adjacency\n", round, wrong);
  }
  static float *k0, *k1; static int *v0, *v1, *prio, *myp, *tm[2], *tgt; static uint8_t *cand, *alive, *matched, *pend; static void* tmp; static size_t tbytes = 0;
  if (!tbytes) {
    CK(cudaMalloc(&k0, 4 * cap)); CK(cudaMalloc(&k1, 4 * cap)); CK(cudaMalloc(&v0, 4 * cap)); CK(cudaMalloc(&v1, 4 * cap));
    CK(cudaMalloc(&prio, 4 * cap)); CK(cudaMalloc(&myp, 4 * cap)); CK(cudaMalloc(&tm[0], 4 * cap)); CK(cudaMalloc(&tm[1], 4 * cap)); CK(cudaMalloc(&tgt, 4 * cap));
    CK(cudaMalloc(&cand, cap)); CK(cudaMalloc(&alive, cap)); CK(cudaMalloc(&matched, cap)); CK(cudaMalloc(&pend, cap));
    cub::DeviceRadixSort::SortPairs(nullptr, tbytes, k0, k1, v0, v1, cap); CK(cudaMalloc(&tmp, tbytes)); }
  const int g = (cap + 255) / 256;
  r_keys<<<g, 256>>>(b.mass, n, cap, k0, v0);
  size_t tb = tbytes; CK(cub::DeviceRadixSort::SortPairs(tmp, tb, k0, k1, v0, v1, cap));
  r_init<<<g, 256>>>(v1, n, R, prio, cand, alive, matched, tgt);
  r_fill<<<g, 256>>>(tm[0], INT_MAX, cap); r_fill<<<g, 256>>>(tm[1], INT_MAX, cap);
  for (int col = 0; col < K; ++col) {
    r_p1<<<g, 256>>>(prio, b.top_idx, cand, alive, matched, n, K, col, tm[col % 2], myp);
    r_p2<<<g, 256>>>(b.top_idx, myp, tm[col % 2], tm[(col + 1) % 2], tgt, matched, alive, n, K, col);
  }
  r_mark<<<g, 256>>>(cand, matched, alive, pend, n);
  std::vector<uint8_t> a1(n), a2(n); std::vector<int> t1(n), t2(n); long long wrong = 0;
  auto cmp8 = [&](const uint8_t* x, const uint8_t* y) { CK(cudaMemcpy(a1.data(), x, n, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(a2.data(), y, n, cudaMemcpyDeviceToHost));
    long long w = 0; for (int i = 0; i < n; ++i) w += (a1[i] != 0) != (a2[i] != 0); return w; };
  const long long wc = cmp8(cand, b.cand), wm = cmp8(matched, b.matched), wa = cmp8(alive, b.alive), wp = cmp8(pend, b.flag);
  CK(cudaMemcpy(t1.data(), tgt, 4 * n, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(t2.data(), b.target, 4 * n, cudaMemcpyDeviceToHost));
  long long wt = 0; for (int i = 0; i < n; ++i) wt += t1[i] != t2[i];
  wrong = wc + wm + wa + wp + wt; g_match_rounds++; g_match_wrong += wrong;
  fprintf(stderr, "    match n=%d remove=%d vs argsort reference: cand %lld matched %lld alive %lld pending %lld target %lld mismatches\n", n, R, wc, wm, wa, wp, wt);
}

int main(int argc, char** argv) {
  std::string dens, rate_path; int reps = 0; int64_t seed = 1234; double nscale = 1.0; int n_force = 0;
  for (int i = 1; i < argc; ++i) { std::string k = argv[i];
    if (k == "--density") dens = argv[++i]; else if (k == "--rate") rate_path = argv[++i];
    else if (k == "--check") g_check = true; else if (k == "--stats") g_stats = true; else if (k == "--jfa") g_jfa = true;
    else if (k == "--reps") reps = atoi(argv[++i]); else if (k == "--seed") seed = atoll(argv[++i]);
    else if (k == "--n") n_force = atoi(argv[++i]); else if (k == "--nscale") nscale = atof(argv[++i]); }
  std::vector<long> sh; auto d = load_npy(dens, &sh); const int H = sh[sh.size() - 2], W = sh.back(), P = H * W; g_H = H; g_W = W;
  double rate = 0; if (!rate_path.empty()) { std::vector<long> s2; rate = load_npy(rate_path, &s2)[0]; }
  const int n_cap = (int)std::min<long long>(1000000, std::max<long long>(1, (long long)(0.40 * P)));
  int n = n_force > 0 ? n_force : std::max(1, std::min((int)(rate >= 1 ? std::min(rate, 2e9) : 1), n_cap));
  n = std::max(1, std::min((int)(n * nscale), n_cap));
  double sum = 0; for (float v : d) sum += v;
  std::vector<float> ds(P); for (int i = 0; i < P; ++i) ds[i] = sum > 1e-8 ? (float)(d[i] * (n / sum)) : d[i];
  gfused::SamplerConfig cfg; cfg.H = H; cfg.W = W; cfg.n_cap = n_cap; cfg.seed = seed;
  gfused::Arena m; gfused::SamplerBuffers sb; gfused::plan_sampler(cfg, m, &sb);
  uint8_t* base; CK(cudaMalloc(&base, m.used())); gfused::Arena bind(base); gfused::plan_sampler(cfg, bind, &sb);
  float* d_w; float2* d_pts; float* d_wout; CK(cudaMalloc(&d_w, P * 4)); CK(cudaMalloc(&d_pts, n_cap * 8)); CK(cudaMalloc(&d_wout, n_cap * 4));
  CK(cudaMemcpy(d_w, d.data(), P * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(sb.density_s, ds.data(), P * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d_bf, P * 4)); CK(cudaMalloc(&d_bfd, P * 4)); CK(cudaMalloc(&d_jo, P * 4)); CK(cudaMalloc(&d_jd, P * 4));
  CK(cudaMalloc(&d_jp, sb.init_cap * 8)); CK(cudaMalloc(&d_cnt, 16)); CK(cudaMalloc(&d_seen, sb.init_cap)); CK(cudaMalloc(&d_lost, sb.init_cap * 4));
  cudaStream_t s; CK(cudaStreamCreate(&s));
  CK(cudaMemcpyAsync(&sb.ctrl->n, &n, 4, cudaMemcpyHostToDevice, s));
  CK(gfused::run_sampler(sb, d_w, d_pts, d_wout, s)); CK(cudaStreamSynchronize(s));
  gfused::SamplerCtrl c; CK(cudaMemcpy(&c, sb.ctrl, sizeof(c), cudaMemcpyDeviceToHost));
  // checksum of outputs
  std::vector<float2> hp(c.n_cur); std::vector<float> hw(c.n_cur);
  CK(cudaMemcpy(hp.data(), d_pts, 8 * c.n_cur, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hw.data(), d_wout, 4 * c.n_cur, cudaMemcpyDeviceToHost));
  double cs = 0; for (int i = 0; i < c.n_cur; ++i) cs += hp[i].x * 3.1 + hp[i].y * 1.7 + hw[i] * 1e-3;
  double ms_med = 0;
  if (reps > 0) {
    g_check = false; cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    CK(cudaMemcpyAsync(&sb.ctrl->n, &n, 4, cudaMemcpyHostToDevice, s));
    CK(gfused::run_sampler(sb, d_w, d_pts, d_wout, s)); CK(cudaStreamEndCapture(s, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); std::vector<float> t;
    for (int r = 0; r < reps + 5; ++r) { cudaEventRecord(e0, s); CK(cudaGraphLaunch(ge, s)); cudaEventRecord(e1, s); CK(cudaEventSynchronize(e1));
      float ms; cudaEventElapsedTime(&ms, e0, e1); if (r >= 5) t.push_back(ms); }
    std::sort(t.begin(), t.end()); ms_med = t[t.size() / 2];
  }
  printf("{\"H\":%d,\"W\":%d,\"n\":%d,\"n_cur\":%d,\"count_error\":%d,\"forced_adj\":%d,\"forced_nn\":%d,\"matched\":%d,\"wsum\":%.3f,\"checksum\":%.6f,"
         "\"adj_wrong\":%lld,\"match_rounds\":%lld,\"match_wrong\":%lld,\"nn_tests\":%lld,\"nn_wrong\":%lld,\"acc_rel_err\":%.3g,\"sampler_ms\":%.4f,\"calls\":%d,\"px\":%lld,\"grid_wrong\":%lld,\"grid_wrong_nontie\":%lld,\"jfa_wrong\":%lld,\"jfa_wrong_nontie\":%lld}\n",
         H, W, n, c.n_cur, c.stat_count_error, c.stat_forced_adj, c.stat_forced_nn, c.stat_matched, c.weight_sum, cs, g_adj_wrong, g_match_rounds, g_match_wrong, g_nn_tests, g_nn_wrong, g_acc_err, ms_med,
         g_acc.calls, g_acc.px, g_acc.wrong, g_acc.wrong_strict, g_acc.jwrong, g_acc.jwrong_strict);
  return 0;
}
