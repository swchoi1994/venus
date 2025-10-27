// Platform detection and SIMD feature detection

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Platform {
    Generic,
    AppleSilicon,
    X86_64,
    ARM64,
    RISCV64,
}

pub fn detect_platform() -> Platform {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    return Platform::AppleSilicon;

    #[cfg(target_arch = "x86_64")]
    return Platform::X86_64;

    #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
    return Platform::ARM64;

    #[cfg(target_arch = "riscv64")]
    return Platform::RISCV64;

    #[allow(unreachable_code)]
    Platform::Generic
}

pub fn get_simd_features() -> Vec<&'static str> {
    let mut features = Vec::new();

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        features.push("NEON");
        features.push("AMX");
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            features.push("AVX");
        }
        if is_x86_feature_detected!("avx2") {
            features.push("AVX2");
        }
        if is_x86_feature_detected!("avx512f") {
            features.push("AVX512");
        }
        if is_x86_feature_detected!("fma") {
            features.push("FMA");
        }
    }

    #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
    {
        features.push("NEON");
        // TODO: Add SVE detection
    }

    if features.is_empty() {
        features.push("None");
    }

    features
}
