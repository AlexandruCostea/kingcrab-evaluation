use serde::Deserialize;

#[derive(Deserialize)]
pub struct EvalEntry {
    pub fen: String,
    pub evals: Vec<EvalInfo>,
}

#[derive(Deserialize)]
pub struct EvalInfo {
    pub depth: u32,
    pub pvs: Vec<PV>,
}

#[derive(Deserialize)]
pub struct PV {
    pub cp: Option<i32>,
    pub mate: Option<i32>,
}
