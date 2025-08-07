use std::env;

fn main() {
    // Tell cargo to look for shared libraries in the project root
    println!("cargo:rustc-link-search=native=.");
    
    // Link to our C library
    println!("cargo:rustc-link-lib=venus");
    
    // Link to system libraries
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=pthread");
    
    // On macOS, link to Accelerate and Metal frameworks
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Metal");
    }
    
    // Rerun if the C library changes
    println!("cargo:rerun-if-changed=libvenus.so");
    println!("cargo:rerun-if-changed=libvenus.dylib");
    println!("cargo:rerun-if-changed=libvenus.a");
}