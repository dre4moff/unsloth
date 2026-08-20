fn main() {
    println!("cargo:rerun-if-env-changed=UNSLOTH_BUNDLED_BACKEND_WHEEL");
    println!("cargo:rerun-if-env-changed=UNSLOTH_BUNDLED_BACKEND_SHA256");
    println!("cargo:rerun-if-env-changed=UNSLOTH_BUNDLED_BACKEND_EXACT_VERSION");
    println!("cargo:rerun-if-env-changed=UNSLOTH_DISABLE_DESKTOP_UPDATES");
    tauri_build::build()
}
