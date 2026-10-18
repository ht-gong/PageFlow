// The TQPlib.tqpkernels module: the operators' CUDA kernels, one source file
// per operator (partition_sort.cu, join_core.cu).

#include <torch/extension.h>

#include <vector>

std::vector<at::Tensor> partition_sort(std::vector<at::Tensor> cols, std::vector<int64_t> comp,
                                       int64_t key_idx, c10::optional<at::Tensor> mask,
                                       int64_t n_valid, int64_t goff, int64_t B,
                                       bool select_first, at::Tensor host_out);
std::vector<at::Tensor> join_expand(at::Tensor sorted_ids, at::Tensor start, at::Tensor len,
                                    at::Tensor host_out);
std::vector<at::Tensor> join_unique_hash(std::vector<at::Tensor> build_keys,
                                         std::vector<at::Tensor> probe_keys,
                                         at::Tensor host_out);
at::Tensor join_semi_anti(at::Tensor build_key, at::Tensor probe_key,
                          c10::optional<at::Tensor> build_val,
                          c10::optional<at::Tensor> probe_val, int64_t op, bool anti,
                          at::Tensor host_out);

PYBIND11_MODULE(tqpkernels, m) {
  m.doc() = "TQP-Vortex operator kernels (operators/kernels)";
  m.def("partition_sort", &partition_sort,
        "Fused stable hash partition of a sort-pipe tick into the padded bucket layout; "
        "waits for the counts (host_out) while the scatter runs",
        pybind11::arg("cols"), pybind11::arg("comp"), pybind11::arg("key_idx"),
        pybind11::arg("mask"), pybind11::arg("n_valid"), pybind11::arg("goff"),
        pybind11::arg("B"), pybind11::arg("select_first"), pybind11::arg("host_out"));
  m.def("join_expand", &join_expand,
        "The (build row, probe row) pairs of each probe row's run in the sorted build; "
        "waits for the pair count (host_out[0])",
        pybind11::arg("sorted_ids"), pybind11::arg("start"), pybind11::arg("len"),
        pybind11::arg("host_out"));
  m.def("join_unique_hash", &join_unique_hash,
        "Inner join on a unique build tuple: (build rows, probe rows) sized for every probe "
        "row, the first host_out[0] the matches; one wait",
        pybind11::arg("build_keys"), pybind11::arg("probe_keys"), pybind11::arg("host_out"));
  m.def("join_semi_anti", &join_semi_anti,
        "Semi or anti join on a hash set of the build keys, with an optional min/max "
        "summary for one residual comparison: the kept probe rows, the first host_out[0]; "
        "one wait",
        pybind11::arg("build_key"), pybind11::arg("probe_key"), pybind11::arg("build_val"),
        pybind11::arg("probe_val"), pybind11::arg("op"), pybind11::arg("anti"),
        pybind11::arg("host_out"));
}
