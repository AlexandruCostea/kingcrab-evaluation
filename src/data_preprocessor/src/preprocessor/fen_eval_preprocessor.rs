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
    file_name_base: String,
    extension: String,
}

impl FenEvalPreprocessor {
    pub fn new(input_path: String, output_path: String, file_name_base: String, extension: String) -> Self {
        Self { input_path, output_path, file_name_base, extension }
    }

    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        let input_file = File::open(&self.input_path)?;
        let reader = Decoder::new(BufReader::new(input_file))?;
        let stream = Deserializer::from_reader(reader).into_iter::<EvalEntry>();


        let mut output_file_name = format!("{}{}1{}", self.output_path, self.file_name_base, self.extension);
        let mut output_file = File::create(&output_file_name)?;
        let mut buf_writer = BufWriter::new(output_file);
        let mut encoder = Encoder::new(buf_writer, 3)?;

        let mut file_index = 1;
        let mut count = 0;
        let mut total_count = 0;

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

            total_count += 1;
            count += 1;

            if count >= 100_000 {
                encoder.flush()?;
                file_index += 1;
                output_file_name = format!("{}{}{}{}", self.output_path, self.file_name_base, file_index, self.extension);
                output_file = File::create(&output_file_name)?;

                buf_writer = BufWriter::new(output_file);
                encoder = Encoder::new(buf_writer, 3)?;

                count = 0;
            }
        }

        encoder.finish()?;


        println!("Wrote {} FEN/eval pairs.", total_count);
        Ok(())
    }
}
