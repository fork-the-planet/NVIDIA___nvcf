// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer};

#[derive(Clone, Debug)]
pub enum LoadBalancerModelConfig {
    Name(LoadBalancerAlgorithm),
    Detailed(Box<LoadBalancerAlgorithmConfig>),
}

impl<'de> Deserialize<'de> for LoadBalancerModelConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        if value.is_string() {
            return serde_json::from_value(value)
                .map(Self::Name)
                .map_err(serde::de::Error::custom);
        }
        if value.is_object() {
            return serde_json::from_value(value)
                .map(|config| Self::Detailed(Box::new(config)))
                .map_err(serde::de::Error::custom);
        }
        Err(serde::de::Error::custom(
            "load-balancer model config must be an algorithm name or detailed config object",
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LoadBalancerAlgorithm {
    PowerOfTwo,
    GroqMultiregion,
    RoundRobin,
    Random,
    Pulsar,
    PulsarMultiregion,
}

impl fmt::Display for LoadBalancerAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::PowerOfTwo => "power-of-two",
            Self::GroqMultiregion => "groq-multiregion",
            Self::RoundRobin => "round-robin",
            Self::Random => "random",
            Self::Pulsar => "pulsar",
            Self::PulsarMultiregion => "pulsar-multiregion",
        };
        write!(f, "{name}")
    }
}

impl FromStr for LoadBalancerAlgorithm {
    type Err = serde::de::value::Error;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Self::deserialize(serde::de::value::StrDeserializer::<serde::de::value::Error>::new(name))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadBalancerAlgorithmOverride {
    raw: String,
    algorithm: LoadBalancerAlgorithm,
}

impl LoadBalancerAlgorithmOverride {
    pub fn parse(value: &str) -> Result<Self, LoadBalancerRoutingAlgorithmError> {
        value.parse()
    }

    pub fn requested_algorithm(&self) -> &str {
        &self.raw
    }

    pub fn algorithm(&self) -> LoadBalancerAlgorithm {
        self.algorithm
    }

    #[cfg(test)]
    pub(super) fn for_test(raw: String, algorithm: LoadBalancerAlgorithm) -> Self {
        Self { raw, algorithm }
    }
}

impl FromStr for LoadBalancerAlgorithmOverride {
    type Err = LoadBalancerRoutingAlgorithmError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let raw = value.trim();
        if raw.is_empty() {
            return Err(LoadBalancerRoutingAlgorithmError::Unknown {
                raw: raw.to_string(),
            });
        }

        let normalized = raw.to_ascii_lowercase();
        let canonical = normalized.replace('_', "-");
        let algorithm =
            canonical
                .parse()
                .map_err(|_| LoadBalancerRoutingAlgorithmError::Unknown {
                    raw: raw.to_string(),
                })?;

        Ok(Self {
            raw: raw.to_string(),
            algorithm,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoadBalancerRoutingAlgorithmError {
    Unknown {
        raw: String,
    },
    Unavailable {
        raw: String,
        algorithm: LoadBalancerAlgorithm,
    },
}

impl LoadBalancerRoutingAlgorithmError {
    pub fn requested_algorithm(&self) -> &str {
        match self {
            Self::Unknown { raw } | Self::Unavailable { raw, .. } => raw,
        }
    }

    pub fn reason(&self) -> &'static str {
        match self {
            Self::Unknown { .. } => "unknown",
            Self::Unavailable { .. } => "unavailable",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoadBalancerSeedError {
    Unsupported { algorithm: LoadBalancerAlgorithm },
}

impl fmt::Display for LoadBalancerSeedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported { algorithm } => write!(f, "seed is not supported for {algorithm}"),
        }
    }
}

impl std::error::Error for LoadBalancerSeedError {}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct LoadBalancerRequestPolicy {
    pub require_cache_affinity_key: bool,
    pub require_input_tokens: bool,
    pub consider_kv_free_tokens: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct GroqMultiregionAlgorithmConfig {
    #[serde(default)]
    pub seed: Option<String>,
    #[serde(default)]
    pub cache_affinity_virtual_nodes: Option<usize>,
    #[serde(default)]
    pub cache_affinity_backend_selection_count: Option<usize>,
    #[serde(default)]
    pub max_queue_time_floor_ms: Option<u64>,
    #[serde(default)]
    pub max_queue_time_ceil_ms: Option<u64>,
    #[serde(default)]
    pub ttft_bucket_size_ms: Option<u64>,
    #[serde(default)]
    pub next_bucket_unlock_factor: Option<f64>,
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub max_queued: Option<u64>,
    #[serde(default)]
    pub ignore_queue_time: Option<bool>,
    #[serde(default)]
    pub ignore_input_processing_time: Option<bool>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LoadBalancerAlgorithmSettings {
    PowerOfTwo,
    GroqMultiregion(GroqMultiregionAlgorithmConfig),
    RoundRobin,
    Random,
    Pulsar(Option<String>),
    PulsarMultiregion(GroqMultiregionAlgorithmConfig),
}

impl LoadBalancerAlgorithmSettings {
    fn algorithm(&self) -> LoadBalancerAlgorithm {
        match self {
            Self::PowerOfTwo => LoadBalancerAlgorithm::PowerOfTwo,
            Self::GroqMultiregion(_) => LoadBalancerAlgorithm::GroqMultiregion,
            Self::RoundRobin => LoadBalancerAlgorithm::RoundRobin,
            Self::Random => LoadBalancerAlgorithm::Random,
            Self::Pulsar(_) => LoadBalancerAlgorithm::Pulsar,
            Self::PulsarMultiregion(_) => LoadBalancerAlgorithm::PulsarMultiregion,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoadBalancerAlgorithmConfig {
    pub request_policy: LoadBalancerRequestPolicy,
    pub max_input_work_seconds: Option<f64>,
    pub request_algorithms: HashMap<LoadBalancerAlgorithm, LoadBalancerModelConfig>,
    pub settings: LoadBalancerAlgorithmSettings,
}

impl LoadBalancerAlgorithmConfig {
    pub fn algorithm(&self) -> LoadBalancerAlgorithm {
        self.settings.algorithm()
    }

    pub fn requires_cache_affinity_key(&self) -> bool {
        self.request_policy.require_cache_affinity_key
    }

    pub fn requires_input_tokens(&self) -> bool {
        self.request_policy.require_input_tokens || self.considers_kv_free_tokens()
    }

    pub fn considers_kv_free_tokens(&self) -> bool {
        self.request_policy.consider_kv_free_tokens
    }

    pub fn request_policy_mut(&mut self) -> &mut LoadBalancerRequestPolicy {
        &mut self.request_policy
    }

    pub fn seed(&self) -> Option<&str> {
        match &self.settings {
            LoadBalancerAlgorithmSettings::GroqMultiregion(config) => config.seed.as_deref(),
            LoadBalancerAlgorithmSettings::Pulsar(seed) => seed.as_deref(),
            LoadBalancerAlgorithmSettings::PulsarMultiregion(config) => config.seed.as_deref(),
            LoadBalancerAlgorithmSettings::PowerOfTwo
            | LoadBalancerAlgorithmSettings::RoundRobin
            | LoadBalancerAlgorithmSettings::Random => None,
        }
    }

    pub fn set_seed(
        &mut self,
        seed: impl Into<Option<String>>,
    ) -> Result<(), LoadBalancerSeedError> {
        let seed = seed.into();
        let algorithm = self.algorithm();
        match &mut self.settings {
            LoadBalancerAlgorithmSettings::GroqMultiregion(config) => {
                config.seed = seed;
                Ok(())
            }
            LoadBalancerAlgorithmSettings::Pulsar(current_seed) => {
                *current_seed = seed;
                Ok(())
            }
            LoadBalancerAlgorithmSettings::PulsarMultiregion(config) => {
                config.seed = seed;
                Ok(())
            }
            LoadBalancerAlgorithmSettings::PowerOfTwo
            | LoadBalancerAlgorithmSettings::RoundRobin
            | LoadBalancerAlgorithmSettings::Random => {
                Err(LoadBalancerSeedError::Unsupported { algorithm })
            }
        }
    }

    pub fn multiregion_settings(&self) -> Option<&GroqMultiregionAlgorithmConfig> {
        match &self.settings {
            LoadBalancerAlgorithmSettings::GroqMultiregion(config) => Some(config),
            LoadBalancerAlgorithmSettings::PulsarMultiregion(config) => Some(config),
            LoadBalancerAlgorithmSettings::PowerOfTwo
            | LoadBalancerAlgorithmSettings::RoundRobin
            | LoadBalancerAlgorithmSettings::Random
            | LoadBalancerAlgorithmSettings::Pulsar(_) => None,
        }
    }

    pub fn multiregion_settings_mut(&mut self) -> Option<&mut GroqMultiregionAlgorithmConfig> {
        match &mut self.settings {
            LoadBalancerAlgorithmSettings::GroqMultiregion(config) => Some(config),
            LoadBalancerAlgorithmSettings::PulsarMultiregion(config) => Some(config),
            LoadBalancerAlgorithmSettings::PowerOfTwo
            | LoadBalancerAlgorithmSettings::RoundRobin
            | LoadBalancerAlgorithmSettings::Random
            | LoadBalancerAlgorithmSettings::Pulsar(_) => None,
        }
    }
}

impl From<LoadBalancerAlgorithm> for LoadBalancerAlgorithmConfig {
    fn from(algorithm: LoadBalancerAlgorithm) -> Self {
        Self {
            request_policy: LoadBalancerRequestPolicy::default(),
            max_input_work_seconds: None,
            request_algorithms: HashMap::new(),
            settings: LoadBalancerAlgorithmSettings::from(algorithm),
        }
    }
}

impl From<LoadBalancerAlgorithm> for LoadBalancerAlgorithmSettings {
    fn from(algorithm: LoadBalancerAlgorithm) -> Self {
        match algorithm {
            LoadBalancerAlgorithm::PowerOfTwo => Self::PowerOfTwo,
            LoadBalancerAlgorithm::GroqMultiregion => {
                Self::GroqMultiregion(GroqMultiregionAlgorithmConfig::default())
            }
            LoadBalancerAlgorithm::RoundRobin => Self::RoundRobin,
            LoadBalancerAlgorithm::Random => Self::Random,
            LoadBalancerAlgorithm::Pulsar => Self::Pulsar(None),
            LoadBalancerAlgorithm::PulsarMultiregion => {
                Self::PulsarMultiregion(GroqMultiregionAlgorithmConfig::default())
            }
        }
    }
}

#[derive(Debug, Default, Deserialize)]
struct RawCommonAlgorithmConfig {
    #[serde(default)]
    require_cache_affinity_key: Option<bool>,
    #[serde(default)]
    require_input_tokens: Option<bool>,
    #[serde(default)]
    max_input_work_seconds: Option<f64>,
    #[serde(default)]
    request_algorithms: HashMap<LoadBalancerAlgorithm, LoadBalancerModelConfig>,
    #[serde(flatten)]
    unsupported_fields: BTreeMap<String, serde_json::Value>,
}

impl RawCommonAlgorithmConfig {
    fn into_config(
        self,
        algorithm: LoadBalancerAlgorithm,
        settings: LoadBalancerAlgorithmSettings,
        consider_kv_free_tokens: Option<bool>,
    ) -> Result<LoadBalancerAlgorithmConfig, String> {
        if !self.unsupported_fields.is_empty() {
            return Err(format!(
                "{} config does not support field(s): {}",
                algorithm,
                self.unsupported_fields
                    .keys()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        let request_policy = LoadBalancerRequestPolicy {
            require_cache_affinity_key: self.require_cache_affinity_key.unwrap_or(false),
            require_input_tokens: self.require_input_tokens.unwrap_or(false),
            consider_kv_free_tokens: consider_kv_free_tokens.unwrap_or(false),
        };
        Ok(LoadBalancerAlgorithmConfig {
            request_policy,
            max_input_work_seconds: self.max_input_work_seconds,
            request_algorithms: self.request_algorithms,
            settings,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "algorithm", rename_all = "kebab-case")]
enum RawLoadBalancerAlgorithmConfig {
    PowerOfTwo {
        #[serde(flatten)]
        common: RawCommonAlgorithmConfig,
    },
    GroqMultiregion {
        #[serde(flatten)]
        settings: GroqMultiregionAlgorithmConfig,
        #[serde(flatten)]
        common: RawCommonAlgorithmConfig,
    },
    RoundRobin {
        #[serde(flatten)]
        common: RawCommonAlgorithmConfig,
    },
    Random {
        #[serde(flatten)]
        common: RawCommonAlgorithmConfig,
    },
    Pulsar {
        #[serde(default)]
        seed: Option<String>,
        #[serde(default)]
        consider_kv_free_tokens: Option<bool>,
        #[serde(flatten)]
        common: RawCommonAlgorithmConfig,
    },
    PulsarMultiregion {
        #[serde(flatten)]
        settings: GroqMultiregionAlgorithmConfig,
        #[serde(default)]
        consider_kv_free_tokens: Option<bool>,
        #[serde(flatten)]
        common: RawCommonAlgorithmConfig,
    },
}

impl RawLoadBalancerAlgorithmConfig {
    fn into_parts(
        self,
    ) -> (
        RawCommonAlgorithmConfig,
        LoadBalancerAlgorithmSettings,
        Option<bool>,
    ) {
        match self {
            Self::PowerOfTwo { common } => {
                (common, LoadBalancerAlgorithmSettings::PowerOfTwo, None)
            }
            Self::GroqMultiregion { settings, common } => (
                common,
                LoadBalancerAlgorithmSettings::GroqMultiregion(settings),
                None,
            ),
            Self::RoundRobin { common } => {
                (common, LoadBalancerAlgorithmSettings::RoundRobin, None)
            }
            Self::Random { common } => (common, LoadBalancerAlgorithmSettings::Random, None),
            Self::Pulsar {
                seed,
                consider_kv_free_tokens,
                common,
            } => (
                common,
                LoadBalancerAlgorithmSettings::Pulsar(seed),
                consider_kv_free_tokens,
            ),
            Self::PulsarMultiregion {
                settings,
                consider_kv_free_tokens,
                common,
            } => (
                common,
                LoadBalancerAlgorithmSettings::PulsarMultiregion(settings),
                consider_kv_free_tokens,
            ),
        }
    }

    fn into_config(self) -> Result<LoadBalancerAlgorithmConfig, String> {
        let (common, settings, consider_kv_free_tokens) = self.into_parts();
        let algorithm = settings.algorithm();
        common.into_config(algorithm, settings, consider_kv_free_tokens)
    }
}

impl<'de> Deserialize<'de> for LoadBalancerAlgorithmConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        RawLoadBalancerAlgorithmConfig::deserialize(deserializer)?
            .into_config()
            .map_err(serde::de::Error::custom)
    }
}

impl LoadBalancerModelConfig {
    pub fn into_algorithm_config(self) -> LoadBalancerAlgorithmConfig {
        match self {
            Self::Name(algorithm) => LoadBalancerAlgorithmConfig::from(algorithm),
            Self::Detailed(config) => *config,
        }
    }
}

fn default_algorithm() -> LoadBalancerAlgorithm {
    LoadBalancerAlgorithm::PowerOfTwo
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoadBalancerConfig {
    #[serde(default = "default_algorithm")]
    pub default: LoadBalancerAlgorithm,
    #[serde(default)]
    pub request_algorithms: HashMap<LoadBalancerAlgorithm, LoadBalancerModelConfig>,
    #[serde(default)]
    pub models: HashMap<String, LoadBalancerModelConfig>,
}

impl Default for LoadBalancerConfig {
    fn default() -> Self {
        Self {
            default: default_algorithm(),
            request_algorithms: HashMap::new(),
            models: HashMap::new(),
        }
    }
}
