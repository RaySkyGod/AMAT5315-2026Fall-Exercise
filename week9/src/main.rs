fn main() {
    let args: Vec<String> = std::env::args().collect();
    eprintln!(
        "infer: probabilistic inference by tensor contraction\n\
         usage: infer --model <path> --evidence <prior|AD|ADX|evid-file> --out <dir>\n\
         args given: {args:?}"
    );
    std::process::exit(2);
}
