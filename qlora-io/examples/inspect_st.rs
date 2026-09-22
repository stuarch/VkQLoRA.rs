use qlora_io::read_safetensors;
fn main() {
    let path = std::env::args().nth(1).expect("usage");
    let bytes = std::fs::read(&path).expect("read");
    let ts = read_safetensors(&bytes).expect("parse");
    let total: usize = ts.iter().map(|t| t.data.len()).sum();
    println!("tensors: {}, params: {}", ts.len(), total);
    for t in ts.iter().take(3) {
        println!(
            "  {} shape={:?} data[0..3]={:?}",
            t.name,
            t.shape,
            &t.data[..3]
        );
    }
    let e = ts
        .iter()
        .find(|t| t.name == "model.embed_tokens.weight")
        .unwrap();
    println!("embed sum: {:.6}", e.data.iter().sum::<f32>());
}
