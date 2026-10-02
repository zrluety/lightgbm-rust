//! Upstream-compatible text model format.
//!
//! upstream: src/boosting/gbdt_model_text.cpp, src/io/tree.cpp.

use crate::boosting::Gbdt;
use crate::dataset::avoid_inf_f64;
use crate::error::{LgbmError, Result};
use crate::fmt::{fmt_g17, parse_f64};
use crate::tree::Tree;

impl Gbdt {
    /// upstream: `GBDT::SaveModelToString`.
    pub fn save_model_to_string(
        &self,
        start_iteration: i32,
        num_iteration: i32,
        importance_type: i32,
    ) -> Result<String> {
        let ntpi = self.num_tree_per_iteration;
        let mut s = String::new();
        s.push_str("tree\n");
        s.push_str("version=v4\n");
        s.push_str(&format!("num_class={}\n", self.num_class));
        s.push_str(&format!("num_tree_per_iteration={ntpi}\n"));
        s.push_str(&format!("label_index={}\n", self.label_index));
        s.push_str(&format!("max_feature_idx={}\n", self.max_feature_idx));
        if let Some(o) = &self.objective {
            s.push_str(&format!("objective={}\n", o.to_model_string()));
        }
        if self.average_output {
            s.push_str("average_output\n");
        }
        s.push_str(&format!("feature_names={}\n", self.feature_names.join(" ")));
        if !self.monotone_constraints.is_empty() {
            s.push_str(&format!("monotone_constraints={}\n", join_i8(&self.monotone_constraints, " ")));
        }
        s.push_str(&format!("feature_infos={}\n", self.feature_infos.join(" ")));

        let mut num_used = self.models.len();
        let total_iter = num_used / ntpi;
        let start_iteration = (start_iteration.max(0) as usize).min(total_iter);
        if num_iteration > 0 {
            num_used = num_used.min((start_iteration + num_iteration as usize) * ntpi);
        }
        let start_model = start_iteration * ntpi;
        let tree_strs: Vec<String> = (start_model..num_used)
            .map(|i| format!("Tree={}\n{}\n", i - start_model, self.models[i].to_model_string()))
            .collect();
        let sizes: Vec<String> = tree_strs.iter().map(|t| t.len().to_string()).collect();
        s.push_str(&format!("tree_sizes={}\n", sizes.join(" ")));
        s.push('\n');
        for t in &tree_strs {
            s.push_str(t);
        }
        s.push_str("end of trees\n");

        let imp = self.feature_importance(num_iteration, importance_type)?;
        let mut pairs: Vec<(usize, &str)> = imp
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| {
                let iv = v as usize;
                (iv > 0).then(|| (iv, self.feature_names[i].as_str()))
            })
            .collect();
        pairs.sort_by(|a, b| b.0.cmp(&a.0)); // stable, like std::stable_sort
        s.push_str("\nfeature_importances:\n");
        for (v, name) in pairs {
            s.push_str(&format!("{name}={v}\n"));
        }
        if let Some(c) = &self.config {
            s.push_str("\nparameters:\n");
            s.push_str(&c.to_model_string());
            s.push('\n');
            s.push_str("end of parameters\n");
        } else if let Some(p) = &self.loaded_parameters {
            s.push_str("\nparameters:\n");
            s.push_str(p);
            s.push('\n');
            s.push_str("end of parameters\n");
        }
        Ok(s)
    }

    /// upstream: `GBDT::DumpModel` (JSON).
    pub fn dump_model(&self, start_iteration: i32, num_iteration: i32, importance_type: i32) -> Result<String> {
        let ntpi = self.num_tree_per_iteration;
        let mut s = String::from("{");
        s.push_str("\"name\":\"tree\",\n\"version\":\"v4\",\n");
        s.push_str(&format!("\"num_class\":{},\n", self.num_class));
        s.push_str(&format!("\"num_tree_per_iteration\":{ntpi},\n"));
        s.push_str(&format!("\"label_index\":{},\n", self.label_index));
        s.push_str(&format!("\"max_feature_idx\":{},\n", self.max_feature_idx));
        if let Some(o) = &self.objective {
            s.push_str(&format!("\"objective\":\"{}\",\n", o.to_model_string()));
        }
        s.push_str(&format!("\"average_output\":{},\n", self.average_output));
        s.push_str(&format!("\"feature_names\":[\"{}\"],\n", self.feature_names.join("\",\"")));
        s.push_str(&format!("\"monotone_constraints\":[{}],\n", join_i8(&self.monotone_constraints, ",")));
        let mut infos = Vec::new();
        for (name, info) in self.feature_names.iter().zip(&self.feature_infos) {
            let Some(range) = info.strip_prefix('[').and_then(|r| r.strip_suffix(']')) else {
                if info != "none" {
                    // categorical: the categories of each bin
                    let vals: Vec<i32> = info
                        .split(':')
                        .map(|v| v.trim().parse::<i32>())
                        .collect::<std::result::Result<_, _>>()
                        .map_err(|_| LgbmError::ModelFormat(format!("bad feature_infos entry {info}")))?;
                    let min = vals.iter().copied().min().unwrap_or(0);
                    let max = vals.iter().copied().max().unwrap_or(0);
                    let joined = vals.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",");
                    infos.push(format!("\"{name}\":{{\"min_value\":{min},\"max_value\":{max},\"values\":[{joined}]}}"));
                }
                continue;
            };
            let (lo, hi) = range
                .split_once(':')
                .ok_or_else(|| LgbmError::ModelFormat(format!("bad feature_infos entry {info}")))?;
            let parse = |v: &str| {
                parse_f64(v).ok_or_else(|| LgbmError::ModelFormat(format!("bad feature_infos entry {info}")))
            };
            infos.push(format!(
                "\"{name}\":{{\"min_value\":{},\"max_value\":{},\"values\":[]}}",
                fmt_g17(avoid_inf_f64(parse(lo)?)),
                fmt_g17(avoid_inf_f64(parse(hi)?))
            ));
        }
        s.push_str(&format!("\"feature_infos\":{{{}}},\n", infos.join(",")));

        let mut num_used = self.models.len();
        let total_iter = num_used / ntpi;
        let start_iteration = (start_iteration.max(0) as usize).min(total_iter);
        if num_iteration > 0 {
            num_used = num_used.min((start_iteration + num_iteration as usize) * ntpi);
        }
        let start_model = start_iteration * ntpi;
        let trees: Vec<String> = (start_model..num_used)
            .map(|i| format!("{{\"tree_index\":{i},{}}}", self.models[i].to_json()))
            .collect();
        s.push_str(&format!("\"tree_info\":[{}],\n", trees.join(",")));

        let imp = self.feature_importance(num_iteration, importance_type)?;
        let pairs: Vec<String> = imp
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v as usize > 0)
            .map(|(i, &v)| format!("\"{}\":{}", self.feature_names[i], v as usize))
            .collect();
        s.push_str(&format!("\n\"feature_importances\":{{{}}}\n}}\n", pairs.join(",")));
        Ok(s)
    }

    /// upstream: `GBDT::LoadModelFromString`. The result can predict and be
    /// saved again but has no training state.
    pub fn load_model_from_string(text: &str) -> Result<Self> {
        let mut kv: std::collections::HashMap<String, String> = Default::default();
        let mut pos = 0usize;
        let bytes = text.as_bytes();
        let next_line = |pos: usize| -> (usize, usize) {
            let end = text[pos..].find(['\n', '\r']).map_or(text.len(), |p| pos + p);
            let mut nxt = end;
            while nxt < bytes.len() && (bytes[nxt] == b'\n' || bytes[nxt] == b'\r') {
                nxt += 1;
            }
            (end, nxt)
        };
        // header
        while pos < text.len() {
            let (end, nxt) = next_line(pos);
            let line = &text[pos..end];
            if line.starts_with("Tree=") {
                break;
            }
            if !line.is_empty() {
                let parts: Vec<&str> = line.split('=').collect();
                match parts.len() {
                    1 => {
                        kv.insert(parts[0].to_string(), String::new());
                    }
                    2 => {
                        kv.insert(parts[0].to_string(), parts[1].to_string());
                    }
                    _ => {
                        if parts[0] == "feature_names" || parts[0] == "monotone_constraints" {
                            kv.insert(parts[0].to_string(), line[parts[0].len() + 1..].to_string());
                        } else {
                            return Err(LgbmError::ModelFormat(format!(
                                "Wrong line at model file: {}",
                                &line[..line.len().min(128)]
                            )));
                        }
                    }
                }
            }
            pos = nxt;
        }
        let get_int = |k: &str| -> Result<i32> {
            kv.get(k)
                .ok_or_else(|| LgbmError::ModelFormat(format!("Model file doesn't specify {k}")))?
                .trim()
                .parse()
                .map_err(|_| LgbmError::ModelFormat(format!("bad {k}")))
        };
        let num_class = get_int("num_class")?;
        let ntpi = if kv.contains_key("num_tree_per_iteration") {
            get_int("num_tree_per_iteration")?
        } else {
            num_class
        };
        if num_class < 1 || ntpi < 1 {
            return Err(LgbmError::ModelFormat("num_class and num_tree_per_iteration must be positive".into()));
        }
        let label_index = get_int("label_index")?;
        let max_feature_idx = get_int("max_feature_idx")?;
        let average_output = kv.contains_key("average_output");
        let feature_names: Vec<String> = kv
            .get("feature_names")
            .ok_or_else(|| LgbmError::ModelFormat("Model file doesn't contain feature_names".into()))?
            .split(' ')
            .map(str::to_string)
            .collect();
        if feature_names.len() != (max_feature_idx + 1) as usize {
            return Err(LgbmError::ModelFormat("Wrong size of feature_names".into()));
        }
        let feature_infos: Vec<String> = kv
            .get("feature_infos")
            .ok_or_else(|| LgbmError::ModelFormat("Model file doesn't contain feature_infos".into()))?
            .split(' ')
            .map(str::to_string)
            .collect();
        if feature_infos.len() != (max_feature_idx + 1) as usize {
            return Err(LgbmError::ModelFormat("Wrong size of feature_infos".into()));
        }
        // upstream: CommonC::StringToArray<int8_t>(..., ' ')
        let monotone_constraints: Vec<i8> = match kv.get("monotone_constraints") {
            Some(v) => {
                let m: Vec<i8> = v.split(' ').filter(|t| !t.is_empty()).map(crate::config::atoi_i8).collect();
                if m.len() != (max_feature_idx + 1) as usize {
                    return Err(LgbmError::ModelFormat("Wrong size of monotone_constraints".into()));
                }
                m
            }
            None => Vec::new(),
        };
        let objective = match kv.get("objective") {
            Some(o) => crate::objective::objective_from_model_string(o)?,
            None => None,
        };

        let mut models = Vec::new();
        if let Some(ts) = kv.get("tree_sizes") {
            let sizes: Vec<usize> = ts
                .split_whitespace()
                .map(|t| t.parse().map_err(|_| LgbmError::ModelFormat("bad tree_sizes".into())))
                .collect::<Result<_>>()?;
            let mut p = pos;
            for sz in sizes {
                if p + sz > text.len() {
                    return Err(LgbmError::ModelFormat("tree_sizes exceed model length".into()));
                }
                let block = &text[p..p + sz];
                let first_end = block.find(['\n', '\r']).unwrap_or(block.len());
                if !block[..first_end].starts_with("Tree=") {
                    return Err(LgbmError::ModelFormat(format!(
                        "Model format error, expect a tree here. met {}",
                        &block[..first_end]
                    )));
                }
                let mut body_start = first_end;
                while body_start < block.len() && matches!(block.as_bytes()[body_start], b'\n' | b'\r') {
                    body_start += 1;
                }
                let (tree, _) = Tree::from_model_str(&block[body_start..])?;
                models.push(tree);
                p += sz;
            }
            pos = p;
        } else {
            while pos < text.len() {
                let (end, nxt) = next_line(pos);
                if text[pos..end].starts_with("Tree=") {
                    let (tree, used) = Tree::from_model_str(&text[nxt..])?;
                    models.push(tree);
                    pos = nxt + used;
                    while pos < text.len() && matches!(bytes[pos], b'\n' | b'\r') {
                        pos += 1;
                    }
                } else {
                    break;
                }
            }
        }
        for t in &models {
            if t.split_feature.iter().any(|&f| f < 0 || f > max_feature_idx) {
                return Err(LgbmError::ModelFormat("split_feature out of range".into()));
            }
        }

        // parameters block
        let mut params = String::new();
        let mut in_params = false;
        while pos < text.len() {
            let (end, nxt) = next_line(pos);
            let line = &text[pos..end];
            if line == "parameters:" {
                in_params = true;
            } else if line == "end of parameters" {
                break;
            } else if in_params {
                params.push_str(line);
                params.push('\n');
            }
            pos = nxt;
        }

        let mut g = Gbdt::empty();
        g.loaded_parameters = if params.is_empty() { None } else { Some(params) };
        g.objective = objective;
        g.models = models;
        g.num_tree_per_iteration = ntpi as usize;
        g.num_class = num_class as usize;
        g.label_index = label_index;
        g.max_feature_idx = max_feature_idx;
        g.feature_names = feature_names;
        g.feature_infos = feature_infos;
        g.monotone_constraints = monotone_constraints;
        g.average_output = average_output;
        Ok(g)
    }
}

/// upstream `Common::Join<int8_t>` (values printed as integers).
fn join_i8(v: &[i8], sep: &str) -> String {
    v.iter().map(|m| m.to_string()).collect::<Vec<_>>().join(sep)
}
