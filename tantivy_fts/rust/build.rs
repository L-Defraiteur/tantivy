fn main() {
    cxx_build::bridge("src/bridge.rs")
        .flag_if_supported("-std=c++17")
        .compile("tantivy_fts_cxx");
}
