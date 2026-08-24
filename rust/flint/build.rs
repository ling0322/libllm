// Points the linker at the static library that the CMake build produces. Override the location
// with LIBLLM_LIB_DIR when building somewhere other than the in-tree `build` directory.
//
// libflint.a carries no record of what it still needs -- libunwind, the CUDA runtime, OpenMP and
// the C++ runtime are all resolved by whoever links it -- and that set depends on the CMake
// options the archive was built with. So CMake writes the whole list out as cargo directives and
// this script echoes them. They are `rustc-link-lib` and `rustc-link-search` rather than raw link
// args on purpose: those two propagate to whatever binary or cdylib ends up linking this crate,
// which is why nothing downstream needs a build script of its own.
use std::path::PathBuf;

fn main() {
    let lib_dir = std::env::var("LIBLLM_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
            manifest.join("../../build")
        });

    let flags_path = lib_dir.join("flint_link_flags.txt");
    let flags = std::fs::read_to_string(&flags_path).unwrap_or_else(|e| {
        panic!(
            "cannot read {}: {e}\n\
             Build the CMake project first (cmake -S . -B build && cmake --build build), or point \
             LIBLLM_LIB_DIR at a directory that has one.",
            flags_path.display()
        )
    });

    println!("cargo:rerun-if-env-changed=LIBLLM_LIB_DIR");
    println!("cargo:rerun-if-changed={}", flags_path.display());
    for line in flags.lines().filter(|l| !l.trim().is_empty()) {
        println!("{line}");
    }
}
