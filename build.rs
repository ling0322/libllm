// build.rs

fn main() {
    let mut lten_build = cc::Build::new();
    lten_build
        .cpp(true)
        .include("cpp")
        .include("third_party")
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-openmp");

    let lten_files = [
        "cpp/lten/cpu/kernel/fallback.cc",
        "cpp/lten/cpu/kernel/interface.cc",
        "cpp/lten/cpu/kernel/util.cc",
        "cpp/lten/cpu/all_close.cc",
        "cpp/lten/cpu/apply_rotary_pos_emb.cc",
        "cpp/lten/cpu/binary_op.cc",
        "cpp/lten/cpu/cast.cc",
        "cpp/lten/cpu/common.cc",
        "cpp/lten/cpu/copy.cc",
        "cpp/lten/cpu/cpu_operators.cc",
        "cpp/lten/cpu/cpu_tensor_data.cc",
        "cpp/lten/cpu/fill.cc",
        "cpp/lten/cpu/fingerprint.cc",
        "cpp/lten/cpu/gelu.cc",
        "cpp/lten/cpu/log_mel_spectrogram.cc",
        "cpp/lten/cpu/lookup.cc",
        "cpp/lten/cpu/matmul.cc",
        "cpp/lten/cpu/normalizations.cc",
        "cpp/lten/cpu/print.cc",
        "cpp/lten/cpu/rand.cc",
        "cpp/lten/cpu/reduce.cc",
        "cpp/lten/cpu/repetition_penalty.cc",
        "cpp/lten/cpu/softmax.cc",
        "cpp/lten/cpu/swiglu.cc",
        "cpp/lten/cpu/tensor.cc",
        "cpp/lten/cpu/transform.cc",
        "cpp/lten/cpu/unfold.cc",
        "cpp/lten/cpu/view.cc",
        "cpp/lten/device.cc",
        "cpp/lten/dtype.cc",
        "cpp/lten/functional.cc",
        "cpp/lten/lten.cc",
        "cpp/lten/mp.cc",
        "cpp/lten/mp_openmp.cc",
        "cpp/lten/operators.cc",
        "cpp/lten/tensor.cc",
    ];
    for file in lten_files.iter() {
        lten_build.file(file);
    }
    lten_build.compile("lten");

    cc::Build::new()
        .cpp(true)
        .include("cpp")
        .include("third_party")
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-mavx2")
        .flag_if_supported("-msse3")
        .flag_if_supported("-mfma")
        .flag_if_supported("-mf16c")
        .file("cpp/lten/cpu/kernel/avx2.cc")
        .compile("lten_avx2");

    cc::Build::new()
        .cpp(true)
        .include("third_party")
        .file("third_party/ruapu/ruapu.cc")
        .compile("ruapu");

    cc::Build::new()
        .cpp(true)
        .include("cpp")
        .include("third_party")
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-mavx512f")
        .file("cpp/lten/cpu/kernel/avx512.cc")
        .compile("lten_avx512");

    let mut lut_build = cc::Build::new();
    lut_build
        .cpp(true)
        .include("cpp")
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-fopenmp");

    let lut_files = [
        "cpp/lutil/internal/log.cc",
        "cpp/lutil/internal/sprintf.cc",
        "cpp/lutil/base64.cc",
        "cpp/lutil/error.cc",
        "cpp/lutil/flags.cc",
        "cpp/lutil/half.cc",
        "cpp/lutil/ini_config.cc",
        "cpp/lutil/is_debug.cc",
        "cpp/lutil/path_linux.cc",
        "cpp/lutil/path.cc",
        "cpp/lutil/platform_linux.cc",
        "cpp/lutil/random.cc",
        "cpp/lutil/reader.cc",
        "cpp/lutil/strings.cc",
        "cpp/lutil/time.cc",
        "cpp/lutil/thread_pool.cc",
        "cpp/lutil/zip_file.cc",
    ];
    for file in lut_files.iter() {
        lut_build.file(file);
    }
    lut_build.compile("lutil");

    println!("cargo:rerun-if-changed=cpp");
    println!("cargo:rustc-link-lib=gomp");
}
