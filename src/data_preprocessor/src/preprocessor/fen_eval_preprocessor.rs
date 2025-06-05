use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use zstd::stream::{read::Decoder, write::Encoder};
use serde::Serialize;
use serde_json::Deserializer;

use super::entry::EvalEntry;

#[derive(Serialize)]
struct FenEval {
    fen: String,
    eval: i32,
}

pub struct FenEvalPreprocessor {
    input_path: String,
    output_path: String,
}

impl FenEvalPreprocessor {
    pub fn new(input_path: String, output_path: String) -> Self {
        Self { input_path, output_path }
    }

    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        let input_file = File::open(&self.input_path)?;
        let reader = Decoder::new(BufReader::new(input_file))?;
        let stream = Deserializer::from_reader(reader).into_iter::<EvalEntry>();

        let output_file = File::create(&self.output_path)?;
        let buf_writer = BufWriter::new(output_file);
        let mut encoder = Encoder::new(buf_writer, 3)?;


        let mut count = 0;

        for result in stream {
            let entry = match result {
                Ok(e) => e,
                Err(_) => continue,
            };

            let best_eval = entry.evals.iter().max_by_key(|e| e.depth);
            let Some(eval) = best_eval else { continue };
            let Some(pv) = eval.pvs.get(0) else { continue };

            let raw_score = pv.cp.or_else(|| pv.mate.map(|m| if m > 0 { 10000 } else { -10000 }));
            let Some(score) = raw_score else { continue };

            let out = FenEval {
                fen: entry.fen,
                eval: score,
            };

            serde_json::to_writer(&mut encoder, &out)?;
            encoder.write_all(b"\n")?;

            count += 1;
        }

        encoder.finish()?;


        println!("Wrote {} FEN/eval pairs.", count);
        Ok(())
    }
}
