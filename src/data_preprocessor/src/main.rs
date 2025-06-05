mod preprocessor;

use preprocessor::fen_eval_preprocessor::FenEvalPreprocessor;

fn main() {
    let unprocessed_data_path = "/home/alexcostea/KingCrab-Evaluation/data/lichess_db_eval.jsonl.zst".to_string();
    let processed_data_path = "/home/alexcostea/KingCrab-Evaluation/data/lichess_db_processed.jsonl.zst".to_string();

    let preprocessor = FenEvalPreprocessor::new(unprocessed_data_path, processed_data_path);

    if let Err(e) = preprocessor.run() {
        eprintln!("Error during preprocessing: {}", e);
    } else {
        println!("Preprocessing completed successfully.");
    }
}
