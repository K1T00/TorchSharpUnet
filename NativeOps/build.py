import os, json, shutil
from pathlib import Path
import torch
import torch.utils.cpp_extension as torch_ext
import setuptools._distutils._msvccompiler as msvc


def get_compute_capabilities():
    
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        cc = major * 10 + minor
        if cc < 75:
            raise RuntimeError("GPUs with compute capability less than 7.5 are not supported.")

    compute_capabilities = {75, 80, 86, 89, 90}
    capability_flags = []
    for cap in compute_capabilities:
        capability_flags += ["-gencode", f"arch=compute_{cap},code=sm_{cap}"]

    return capability_flags

def append_env(env: str, paths: str):
    paths = paths.split(os.pathsep)
    for p in paths:
        if env not in os.environ:
            os.environ[env] = ""
        if len(p) and p not in os.environ[env]:
            os.environ[env] = p + os.pathsep + os.environ[env]



######### START ############



build_name = "TorchOps"
root = Path(__file__).parent

build_path = root / "build"
build_path.mkdir(parents=True, exist_ok=True)

sources = [root / "src/nativeops.cpp"]

extra_cflags = []
extra_cuda_cflags = []
extra_ld_flags = []
extra_include_paths = []

print(json.dumps(torch_ext.include_paths(cuda=True), indent=2))
print(json.dumps(torch_ext.library_paths(cuda=True), indent=2))

vc_env = msvc._get_vc_env("x86_amd64")

append_env("path", vc_env.get("path", ""))
append_env("include", vc_env.get("include", ""))
append_env("lib", vc_env.get("lib", ""))

extra_cflags += [
    "/Ox", "/std:c++17"
]
extra_cuda_cflags += [
    "-O3", 
    "-std=c++17",
    "-DUSE_NVTX=ON",
    # "-DENABLE_BF16",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "--use_fast_math",
]


extra_cuda_cflags += get_compute_capabilities()



torch_ext.load(
    build_name,
    sources=[str(s) for s in sources],
    build_directory=str(build_path),
    extra_cflags=extra_cflags,
    extra_cuda_cflags=extra_cuda_cflags,
    extra_ldflags=extra_ld_flags,
    extra_include_paths=extra_include_paths,
    with_cuda=True,
    is_python_module=False,
    verbose=True
)

runtimes_path = root / "runtimes"


build_target = build_path / (build_name + torch_ext.LIB_EXT)
runtimes_target = runtimes_path / "win-x64" / "native" / (build_name + torch_ext.CLIB_EXT)
runtimes_target.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(build_target, runtimes_target)



print("Build success")
print("Copied output file to", runtimes_target)
