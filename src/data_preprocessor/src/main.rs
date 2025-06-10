mod preprocessor;

use preprocessor::fen_eval_preprocessor::FenEvalPreprocessor;

use std::env;
use std::process;


fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <unprocessed_data_path> <processed_data_path>", args[0]);
        process::exit(1);
    }

    let unprocessed_data_path = args[1].clone();
    let processed_data_path = args[2].clone();

    let preprocessor = FenEvalPreprocessor::new(unprocessed_data_path, processed_data_path);

    if let Err(e) = preprocessor.run() {
        eprintln!("Error during preprocessing: {}", e);
    } else {
        println!("Preprocessing completed successfully.");
    }
}