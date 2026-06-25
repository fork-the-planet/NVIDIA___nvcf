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

use std::borrow::Cow;
use std::fmt;
use std::marker::PhantomData;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use futures::{Stream, StreamExt};
use reqwest::StatusCode;
use serde::de::{self, Deserialize, Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};
use tokio::time::Instant as TokioInstant;
use tokio_util::sync::CancellationToken;

use stargate_runtime::{OwnedTask, TASK_SHUTDOWN_TIMEOUT};

use super::collector::{RequestCounterUpdate, StatsAggregatorUpdate, StatsUpdateSource};
use super::metrics::PylonMetrics;

const DEFAULT_ENGINE_STATS_STREAM_PATH: &str = "/pylon/v1/stats/stream";
const DEFAULT_INITIAL_RECONNECT_BACKOFF: Duration = Duration::from_millis(100);
const DEFAULT_MAX_RECONNECT_BACKOFF: Duration = Duration::from_secs(5);
const DEFAULT_MAX_LINE_BYTES: usize = 64 * 1024;
const HEADER_ACCEPT_NDJSON: &str = "application/x-ndjson";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineStatsStreamMode {
    Auto,
    Required,
    Off,
}

impl EngineStatsStreamMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Required => "required",
            Self::Off => "off",
        }
    }
}

impl fmt::Display for EngineStatsStreamMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for EngineStatsStreamMode {
    type Err = ParseEngineStatsStreamModeError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "required" => Ok(Self::Required),
            "off" => Ok(Self::Off),
            _ => Err(ParseEngineStatsStreamModeError),
        }
    }
}

#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("expected one of auto, required, off")]
pub struct ParseEngineStatsStreamModeError;

#[derive(Debug, Clone)]
pub struct EngineStatsStreamConfig {
    pub url: String,
    pub mode: EngineStatsStreamMode,
    pub initial_reconnect_backoff: Duration,
    pub max_reconnect_backoff: Duration,
    pub max_line_bytes: usize,
    pub metrics: Option<Arc<PylonMetrics>>,
}

impl EngineStatsStreamConfig {
    pub fn new(upstream_base_url: &str, path: &str, mode: EngineStatsStreamMode) -> Self {
        Self {
            url: join_base_url_path(upstream_base_url, path),
            mode,
            initial_reconnect_backoff: DEFAULT_INITIAL_RECONNECT_BACKOFF,
            max_reconnect_backoff: DEFAULT_MAX_RECONNECT_BACKOFF,
            max_line_bytes: DEFAULT_MAX_LINE_BYTES,
            metrics: None,
        }
    }
}

impl Default for EngineStatsStreamConfig {
    fn default() -> Self {
        Self::new(
            "http://127.0.0.1:8090",
            DEFAULT_ENGINE_STATS_STREAM_PATH,
            EngineStatsStreamMode::Auto,
        )
    }
}

pub struct EngineStatsStreamHandle {
    task: OwnedTask,
}

impl EngineStatsStreamHandle {
    pub async fn wait_for_exit(&mut self) -> Result<(), tokio::task::JoinError> {
        self.task.wait_for_exit().await
    }

    pub async fn shutdown(self) {
        self.task.shutdown(TASK_SHUTDOWN_TIMEOUT).await;
    }
}

pub fn start_engine_stats_stream(
    config: EngineStatsStreamConfig,
    stats_update_tx: flume::Sender<StatsAggregatorUpdate>,
) -> Option<EngineStatsStreamHandle> {
    if config.mode == EngineStatsStreamMode::Off {
        return None;
    }
    let task = OwnedTask::spawn("engine stats stream", move |stop| {
        run_engine_stats_stream(config, stats_update_tx, stop)
    });
    Some(EngineStatsStreamHandle { task })
}

#[derive(Debug, Clone)]
pub(crate) enum ParsedEngineStatsEvent {
    Update(StatsAggregatorUpdate),
    Ping,
}

impl ParsedEngineStatsEvent {
    fn event_type(&self) -> &'static str {
        match self {
            Self::Update(StatsAggregatorUpdate::RequestCounters(_)) => "stats",
            Self::Update(StatsAggregatorUpdate::FinalizeRequest(_)) => "finalize",
            Self::Update(StatsAggregatorUpdate::EnableOpenAiFallback) => "control",
            Self::Ping => "ping",
        }
    }

    fn into_update(self) -> Option<StatsAggregatorUpdate> {
        match self {
            Self::Update(update) => Some(update),
            Self::Ping => None,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum EngineStatsParseError {
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("event must be a JSON object")]
    NotObject,
    #[error("missing field {0}")]
    MissingField(&'static str),
    #[error("unsupported version {0}")]
    UnsupportedVersion(u64),
    #[error("invalid field {0}")]
    InvalidField(&'static str),
    #[error("unknown event type {0}")]
    UnknownType(String),
    #[error("stats event must include at least one counter unless finished=true")]
    EmptyStatsCounters,
}

pub(crate) fn parse_engine_stats_line(
    line: &[u8],
    observed_at: TokioInstant,
) -> Result<ParsedEngineStatsEvent, EngineStatsParseError> {
    let raw: RawEngineStatsLine<'_> = serde_json::from_slice(line)?;
    let RawEngineStatsLine::Object(mut raw) = raw else {
        return Err(EngineStatsParseError::NotObject);
    };
    let version = required_u64(raw.version.take(), "v")?;
    if version != 1 {
        return Err(EngineStatsParseError::UnsupportedVersion(version));
    }
    let event_type = required_str(raw.event_type.as_ref(), "type")?;
    if event_type == "stats" {
        parse_stats_event(raw, observed_at)
    } else if event_type == "ping" {
        Ok(ParsedEngineStatsEvent::Ping)
    } else {
        Err(EngineStatsParseError::UnknownType(event_type.to_string()))
    }
}

#[doc(hidden)]
pub fn parse_engine_stats_line_for_benchmark(line: &[u8], observed_at: TokioInstant) -> bool {
    parse_engine_stats_line(line, observed_at).is_ok()
}

fn parse_stats_event(
    raw: RawEngineStatsEvent<'_>,
    observed_at: TokioInstant,
) -> Result<ParsedEngineStatsEvent, EngineStatsParseError> {
    let request_id = required_nonempty_string(raw.request_id, "request_id")?;
    let model_id = required_nonempty_string(raw.model, "model")?;
    let tokens_processed = optional_u64(raw.tokens_processed, "tokens_processed")?;
    let tokens_generated = optional_u64(raw.tokens_generated, "tokens_generated")?;
    let finished = optional_bool(raw.finished, "finished")?.unwrap_or(false);
    if tokens_processed.is_none() && tokens_generated.is_none() && !finished {
        return Err(EngineStatsParseError::EmptyStatsCounters);
    }
    Ok(ParsedEngineStatsEvent::Update(
        StatsAggregatorUpdate::RequestCounters(RequestCounterUpdate {
            source: StatsUpdateSource::EngineStatsStream,
            request_id,
            model_id,
            tokens_processed,
            tokens_generated,
            finished,
            observed_at,
        }),
    ))
}

enum RawEngineStatsLine<'a> {
    Object(RawEngineStatsEvent<'a>),
    NotObject,
}

#[derive(Default)]
struct RawEngineStatsEvent<'a> {
    version: Option<JsonU64Field>,
    event_type: Option<JsonStringField<'a>>,
    request_id: Option<JsonStringField<'a>>,
    model: Option<JsonStringField<'a>>,
    tokens_processed: Option<JsonU64Field>,
    tokens_generated: Option<JsonU64Field>,
    finished: Option<JsonBoolField>,
}

impl<'de> Deserialize<'de> for RawEngineStatsLine<'de> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(RawEngineStatsLineVisitor)
    }
}

struct RawEngineStatsLineVisitor;

impl<'de> Visitor<'de> for RawEngineStatsLineVisitor {
    type Value = RawEngineStatsLine<'de>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an engine stats JSON object")
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut event = RawEngineStatsEvent::default();
        while let Some(key) = map.next_key::<Cow<'de, str>>()? {
            match key.as_ref() {
                "v" => event.version = Some(map.next_value()?),
                "type" => event.event_type = Some(map.next_value()?),
                "request_id" => event.request_id = Some(map.next_value()?),
                "model" => event.model = Some(map.next_value()?),
                "tokens_processed" => event.tokens_processed = Some(map.next_value()?),
                "tokens_generated" => event.tokens_generated = Some(map.next_value()?),
                "finished" => event.finished = Some(map.next_value()?),
                _ => {
                    let _ = map.next_value::<IgnoredAny>()?;
                }
            }
        }
        Ok(RawEngineStatsLine::Object(event))
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        while seq.next_element::<IgnoredAny>()?.is_some() {}
        Ok(RawEngineStatsLine::NotObject)
    }

    fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(RawEngineStatsLine::NotObject)
    }

    fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(RawEngineStatsLine::NotObject)
    }

    fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(RawEngineStatsLine::NotObject)
    }

    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(RawEngineStatsLine::NotObject)
    }

    fn visit_str<E>(self, _value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(RawEngineStatsLine::NotObject)
    }

    fn visit_borrowed_str<E>(self, _value: &'de str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(RawEngineStatsLine::NotObject)
    }

    fn visit_string<E>(self, _value: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(RawEngineStatsLine::NotObject)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(RawEngineStatsLine::NotObject)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(RawEngineStatsLine::NotObject)
    }
}

#[derive(Debug)]
enum JsonStringField<'a> {
    Value(Cow<'a, str>),
    Invalid,
}

impl<'de> Deserialize<'de> for JsonStringField<'de> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(JsonStringFieldVisitor(PhantomData))
    }
}

struct JsonStringFieldVisitor<'a>(PhantomData<&'a ()>);

impl<'de> Visitor<'de> for JsonStringFieldVisitor<'de> {
    type Value = JsonStringField<'de>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON string")
    }

    fn visit_borrowed_str<E>(self, value: &'de str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonStringField::Value(Cow::Borrowed(value)))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonStringField::Value(Cow::Owned(value.to_string())))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonStringField::Value(Cow::Owned(value)))
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        while seq.next_element::<IgnoredAny>()?.is_some() {}
        Ok(JsonStringField::Invalid)
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
        Ok(JsonStringField::Invalid)
    }

    fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonStringField::Invalid)
    }

    fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonStringField::Invalid)
    }

    fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonStringField::Invalid)
    }

    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonStringField::Invalid)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonStringField::Invalid)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonStringField::Invalid)
    }
}

#[derive(Debug)]
enum JsonU64Field {
    Value(u64),
    Invalid,
}

impl<'de> Deserialize<'de> for JsonU64Field {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(JsonU64FieldVisitor)
    }
}

struct JsonU64FieldVisitor;

impl<'de> Visitor<'de> for JsonU64FieldVisitor {
    type Value = JsonU64Field;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a non-negative JSON integer")
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonU64Field::Value(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(u64::try_from(value)
            .map(JsonU64Field::Value)
            .unwrap_or(JsonU64Field::Invalid))
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        while seq.next_element::<IgnoredAny>()?.is_some() {}
        Ok(JsonU64Field::Invalid)
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
        Ok(JsonU64Field::Invalid)
    }

    fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonU64Field::Invalid)
    }

    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonU64Field::Invalid)
    }

    fn visit_str<E>(self, _value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonU64Field::Invalid)
    }

    fn visit_borrowed_str<E>(self, _value: &'de str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonU64Field::Invalid)
    }

    fn visit_string<E>(self, _value: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonU64Field::Invalid)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonU64Field::Invalid)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonU64Field::Invalid)
    }
}

#[derive(Debug)]
enum JsonBoolField {
    Value(bool),
    Invalid,
}

impl<'de> Deserialize<'de> for JsonBoolField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(JsonBoolFieldVisitor)
    }
}

struct JsonBoolFieldVisitor;

impl<'de> Visitor<'de> for JsonBoolFieldVisitor {
    type Value = JsonBoolField;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON boolean")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonBoolField::Value(value))
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        while seq.next_element::<IgnoredAny>()?.is_some() {}
        Ok(JsonBoolField::Invalid)
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
        Ok(JsonBoolField::Invalid)
    }

    fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonBoolField::Invalid)
    }

    fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonBoolField::Invalid)
    }

    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonBoolField::Invalid)
    }

    fn visit_str<E>(self, _value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonBoolField::Invalid)
    }

    fn visit_borrowed_str<E>(self, _value: &'de str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonBoolField::Invalid)
    }

    fn visit_string<E>(self, _value: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonBoolField::Invalid)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonBoolField::Invalid)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(JsonBoolField::Invalid)
    }
}

fn required_u64(
    value: Option<JsonU64Field>,
    field: &'static str,
) -> Result<u64, EngineStatsParseError> {
    optional_u64(value, field)?.ok_or(EngineStatsParseError::MissingField(field))
}

fn optional_u64(
    value: Option<JsonU64Field>,
    field: &'static str,
) -> Result<Option<u64>, EngineStatsParseError> {
    match value {
        Some(JsonU64Field::Value(value)) => Ok(Some(value)),
        Some(JsonU64Field::Invalid) => Err(EngineStatsParseError::InvalidField(field)),
        None => Ok(None),
    }
}

fn optional_bool(
    value: Option<JsonBoolField>,
    field: &'static str,
) -> Result<Option<bool>, EngineStatsParseError> {
    match value {
        Some(JsonBoolField::Value(value)) => Ok(Some(value)),
        Some(JsonBoolField::Invalid) => Err(EngineStatsParseError::InvalidField(field)),
        None => Ok(None),
    }
}

fn required_str<'a>(
    value: Option<&'a JsonStringField<'a>>,
    field: &'static str,
) -> Result<&'a str, EngineStatsParseError> {
    match value {
        Some(JsonStringField::Value(value)) => Ok(value.as_ref()),
        Some(JsonStringField::Invalid) => Err(EngineStatsParseError::InvalidField(field)),
        None => Err(EngineStatsParseError::MissingField(field)),
    }
}

fn required_nonempty_string(
    value: Option<JsonStringField<'_>>,
    field: &'static str,
) -> Result<String, EngineStatsParseError> {
    let value = match value {
        Some(JsonStringField::Value(value)) => value,
        Some(JsonStringField::Invalid) => return Err(EngineStatsParseError::InvalidField(field)),
        None => return Err(EngineStatsParseError::MissingField(field)),
    };
    let value = value.trim();
    if value.is_empty() {
        return Err(EngineStatsParseError::InvalidField(field));
    }
    Ok(value.to_string())
}

async fn run_engine_stats_stream(
    config: EngineStatsStreamConfig,
    stats_update_tx: flume::Sender<StatsAggregatorUpdate>,
    stop: CancellationToken,
) {
    let client = reqwest::Client::new();
    let mut backoff = config.initial_reconnect_backoff;
    let mut valid_event_seen = false;
    loop {
        if stop.is_cancelled() {
            return;
        }
        match read_stream_once(
            &config,
            &client,
            &stats_update_tx,
            &stop,
            &mut valid_event_seen,
        )
        .await
        {
            StreamReadOutcome::Stopped => return,
            StreamReadOutcome::Unsupported
                if config.mode == EngineStatsStreamMode::Auto && !valid_event_seen =>
            {
                tracing::warn!(
                    url = config.url,
                    "engine stats stream unsupported; using OpenAI fallback observation"
                );
                let _ = send_stats_update(
                    &stats_update_tx,
                    StatsAggregatorUpdate::EnableOpenAiFallback,
                    &stop,
                )
                .await;
                return;
            }
            StreamReadOutcome::Unsupported => {
                observe_reconnect(&config, "unsupported");
            }
            StreamReadOutcome::Retry(reason) => {
                observe_reconnect(&config, reason);
            }
        }

        if sleep_or_stop(backoff, &stop).await {
            return;
        }
        backoff = (backoff * 2).min(config.max_reconnect_backoff);
    }
}

#[derive(Debug, PartialEq, Eq)]
enum StreamReadOutcome {
    Stopped,
    Unsupported,
    Retry(&'static str),
}

async fn read_stream_once(
    config: &EngineStatsStreamConfig,
    client: &reqwest::Client,
    stats_update_tx: &flume::Sender<StatsAggregatorUpdate>,
    stop: &CancellationToken,
    valid_event_seen: &mut bool,
) -> StreamReadOutcome {
    let response = match open_engine_stats_response(config, client, stop).await {
        Ok(response) => response,
        Err(outcome) => return outcome,
    };
    observe_connected(config, true);
    drain_engine_stats_response(config, response, stats_update_tx, stop, valid_event_seen).await
}

async fn open_engine_stats_response(
    config: &EngineStatsStreamConfig,
    client: &reqwest::Client,
    stop: &CancellationToken,
) -> Result<reqwest::Response, StreamReadOutcome> {
    let response = tokio::select! {
        _ = stop.cancelled() => return Err(StreamReadOutcome::Stopped),
        response = client
            .get(&config.url)
            .header(reqwest::header::ACCEPT, HEADER_ACCEPT_NDJSON)
            .send() => response,
    };
    let response = match response {
        Ok(response) => response,
        Err(error) => {
            tracing::warn!(url = config.url, error = %error, "engine stats stream connect failed");
            return Err(StreamReadOutcome::Retry("connect_error"));
        }
    };
    classify_engine_stats_response(config, response)
}

fn classify_engine_stats_response(
    config: &EngineStatsStreamConfig,
    response: reqwest::Response,
) -> Result<reqwest::Response, StreamReadOutcome> {
    if permanent_unsupported_status(response.status()) {
        tracing::warn!(
            url = config.url,
            status = %response.status(),
            "engine stats stream endpoint is unsupported"
        );
        return Err(StreamReadOutcome::Unsupported);
    }
    if !response.status().is_success() {
        tracing::warn!(
            url = config.url,
            status = %response.status(),
            "engine stats stream returned non-success status"
        );
        return Err(StreamReadOutcome::Retry("http_status"));
    }
    Ok(response)
}

async fn drain_engine_stats_response(
    config: &EngineStatsStreamConfig,
    response: reqwest::Response,
    stats_update_tx: &flume::Sender<StatsAggregatorUpdate>,
    stop: &CancellationToken,
    valid_event_seen: &mut bool,
) -> StreamReadOutcome {
    let mut stream = response.bytes_stream();
    let mut line_buffer = Vec::with_capacity(1024);
    loop {
        let chunk = match next_engine_stats_chunk(config, stop, &mut stream).await {
            Ok(chunk) => match non_empty_engine_stats_chunk(config, chunk) {
                Ok(chunk) => chunk,
                Err(outcome) => return outcome,
            },
            Err(outcome) => return outcome,
        };
        if let Some(outcome) = process_engine_stats_chunk(
            config,
            stats_update_tx,
            valid_event_seen,
            &mut line_buffer,
            &chunk,
            stop,
        )
        .await
        {
            return outcome;
        }
    }
}

fn non_empty_engine_stats_chunk(
    config: &EngineStatsStreamConfig,
    chunk: Bytes,
) -> Result<Bytes, StreamReadOutcome> {
    if chunk.is_empty() {
        observe_connected(config, false);
        Err(StreamReadOutcome::Retry("empty_chunk"))
    } else {
        Ok(chunk)
    }
}

async fn next_engine_stats_chunk<S>(
    config: &EngineStatsStreamConfig,
    stop: &CancellationToken,
    stream: &mut S,
) -> Result<Bytes, StreamReadOutcome>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    let chunk = tokio::select! {
        _ = stop.cancelled() => {
            observe_connected(config, false);
            return Err(StreamReadOutcome::Stopped);
        }
        chunk = stream.next() => chunk,
    };
    let Some(chunk) = chunk else {
        observe_connected(config, false);
        return Err(StreamReadOutcome::Retry("eof"));
    };
    chunk.map_err(|error| {
        tracing::warn!(url = config.url, error = %error, "engine stats stream read failed");
        observe_connected(config, false);
        StreamReadOutcome::Retry("read_error")
    })
}

async fn process_engine_stats_chunk(
    config: &EngineStatsStreamConfig,
    stats_update_tx: &flume::Sender<StatsAggregatorUpdate>,
    valid_event_seen: &mut bool,
    line_buffer: &mut Vec<u8>,
    chunk: &[u8],
    stop: &CancellationToken,
) -> Option<StreamReadOutcome> {
    line_buffer.extend_from_slice(chunk);
    process_buffered_engine_stats_lines(
        config,
        stats_update_tx,
        valid_event_seen,
        line_buffer,
        stop,
    )
    .await
}

async fn process_buffered_engine_stats_lines(
    config: &EngineStatsStreamConfig,
    stats_update_tx: &flume::Sender<StatsAggregatorUpdate>,
    valid_event_seen: &mut bool,
    line_buffer: &mut Vec<u8>,
    stop: &CancellationToken,
) -> Option<StreamReadOutcome> {
    let mut consumed = 0;
    while let Some(relative_newline_index) = memchr::memchr(b'\n', &line_buffer[consumed..]) {
        let newline_index = consumed + relative_newline_index;
        if newline_index - consumed > config.max_line_bytes {
            observe_invalid(config, "line_too_large");
            consumed = newline_index.saturating_add(1);
            continue;
        }
        let line_end = trim_engine_stats_line_end(line_buffer, consumed, newline_index);
        if line_end != consumed {
            let parsed_event = {
                let line = &line_buffer[consumed..line_end];
                parse_engine_stats_line(line, TokioInstant::now())
            };
            if !handle_parsed_engine_stats_event(
                config,
                stats_update_tx,
                valid_event_seen,
                parsed_event,
                stop,
            )
            .await
            {
                observe_connected(config, false);
                return Some(StreamReadOutcome::Stopped);
            }
        }
        consumed = newline_index.saturating_add(1);
    }
    compact_engine_stats_line_buffer(config, line_buffer, consumed);
    None
}

async fn handle_parsed_engine_stats_event(
    config: &EngineStatsStreamConfig,
    stats_update_tx: &flume::Sender<StatsAggregatorUpdate>,
    valid_event_seen: &mut bool,
    parsed_event: Result<ParsedEngineStatsEvent, EngineStatsParseError>,
    stop: &CancellationToken,
) -> bool {
    match parsed_event {
        Ok(event) => {
            *valid_event_seen = true;
            observe_event(config, event.event_type());
            match event.into_update() {
                Some(update) => send_stats_update(stats_update_tx, update, stop).await,
                None => true,
            }
        }
        Err(error) => {
            tracing::warn!(url = config.url, error = %error, "invalid engine stats event");
            observe_invalid(config, error.metric_reason());
            true
        }
    }
}

fn trim_engine_stats_line_end(line_buffer: &[u8], consumed: usize, newline_index: usize) -> usize {
    let mut line_end = newline_index;
    if line_buffer
        .get(consumed..line_end)
        .is_some_and(|line| line.ends_with(b"\r"))
    {
        line_end = line_end.saturating_sub(1);
    }
    line_end
}

fn compact_engine_stats_line_buffer(
    config: &EngineStatsStreamConfig,
    line_buffer: &mut Vec<u8>,
    consumed: usize,
) {
    if consumed == line_buffer.len() {
        line_buffer.clear();
    } else {
        let _ = line_buffer.drain(..consumed).count();
    }
    if line_buffer.len() > config.max_line_bytes {
        observe_invalid(config, "line_too_large");
        line_buffer.clear();
    }
}

async fn send_stats_update(
    stats_update_tx: &flume::Sender<StatsAggregatorUpdate>,
    update: StatsAggregatorUpdate,
    stop: &CancellationToken,
) -> bool {
    match stats_update_tx.try_send(update) {
        Ok(()) => true,
        Err(flume::TrySendError::Full(update)) => stop
            .run_until_cancelled(stats_update_tx.send_async(update))
            .await
            .is_some_and(|result| result.is_ok()),
        Err(flume::TrySendError::Disconnected(_)) => false,
    }
}

impl EngineStatsParseError {
    fn metric_reason(&self) -> &'static str {
        match self {
            Self::Json(_) => "json",
            Self::NotObject => "not_object",
            Self::MissingField(_) => "missing_field",
            Self::UnsupportedVersion(_) => "version",
            Self::InvalidField(_) => "field",
            Self::UnknownType(_) => "type",
            Self::EmptyStatsCounters => "empty_stats",
        }
    }
}

fn observe_event(config: &EngineStatsStreamConfig, event_type: &'static str) {
    if let Some(metrics) = &config.metrics {
        metrics.observe_engine_stats_stream_event(event_type);
    }
}

fn observe_invalid(config: &EngineStatsStreamConfig, reason: &'static str) {
    if let Some(metrics) = &config.metrics {
        metrics.observe_engine_stats_invalid_event(reason);
    }
}

fn observe_reconnect(config: &EngineStatsStreamConfig, reason: &'static str) {
    if let Some(metrics) = &config.metrics {
        metrics.observe_engine_stats_reconnect(reason);
    }
}

fn observe_connected(config: &EngineStatsStreamConfig, connected: bool) {
    if let Some(metrics) = &config.metrics {
        metrics.observe_engine_stats_stream_connected(config.mode.as_str(), connected);
    }
}

fn permanent_unsupported_status(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::NOT_FOUND | StatusCode::METHOD_NOT_ALLOWED | StatusCode::NOT_IMPLEMENTED
    )
}

async fn sleep_or_stop(duration: Duration, stop: &CancellationToken) -> bool {
    stop.run_until_cancelled(tokio::time::sleep(duration))
        .await
        .is_none()
}

fn join_base_url_path(base_url: &str, path: &str) -> String {
    format!(
        "{}/{}",
        base_url.trim_end_matches('/'),
        path.trim_start_matches('/')
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, routing::get};
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use tokio::net::TcpListener;

    fn parse(line: &[u8]) -> Result<ParsedEngineStatsEvent, EngineStatsParseError> {
        parse_engine_stats_line(line, TokioInstant::now())
    }

    #[test]
    fn parses_valid_engine_stats_events() {
        let event = parse(
            br#"{"v":1,"type":"stats","request_id":"req-1","model":"llama","tokens_processed":4096,"tokens_generated":128,"finished":true}"#,
        )
        .expect("stats event should parse");
        let ParsedEngineStatsEvent::Update(StatsAggregatorUpdate::RequestCounters(update)) = event
        else {
            panic!("expected request counters update");
        };
        assert_eq!(update.request_id, "req-1");
        assert_eq!(update.model_id, "llama");
        assert_eq!(update.tokens_processed, Some(4096));
        assert_eq!(update.tokens_generated, Some(128));
        assert!(update.finished);

        assert!(matches!(
            parse(br#"{"v":1,"type":"ping"}"#).expect("ping should parse"),
            ParsedEngineStatsEvent::Ping
        ));
    }

    #[test]
    fn rejects_invalid_engine_stats_events() {
        assert!(matches!(
            parse(br#"{"v":2,"type":"ping"}"#).unwrap_err(),
            EngineStatsParseError::UnsupportedVersion(2)
        ));
        assert!(matches!(
            parse(br#"{"v":1,"type":"nope"}"#).unwrap_err(),
            EngineStatsParseError::UnknownType(_)
        ));
        assert!(matches!(
            parse(br#"{"v":1,"type":"stats","request_id":"req-1","model":"llama"}"#).unwrap_err(),
            EngineStatsParseError::EmptyStatsCounters
        ));
        assert!(matches!(
            parse(
                br#"{"v":1,"type":"stats","request_id":"req-1","model":"llama","tokens_processed":-1}"#,
            )
            .unwrap_err(),
            EngineStatsParseError::InvalidField("tokens_processed")
        ));
        assert!(matches!(
            parse(
                br#"{"v":1,"type":"stats","request_id":"req-1","model":"llama","tokens_generated":1.5}"#,
            )
            .unwrap_err(),
            EngineStatsParseError::InvalidField("tokens_generated")
        ));
    }

    #[tokio::test]
    async fn next_engine_stats_chunk_reads_before_cancellation() {
        let config = EngineStatsStreamConfig::default();
        let stop = CancellationToken::new();
        let (chunk_tx, chunk_rx) = tokio::sync::mpsc::channel(1);
        let reader = tokio::spawn(async move {
            let mut stream = tokio_stream::wrappers::ReceiverStream::new(chunk_rx);
            next_engine_stats_chunk(&config, &stop, &mut stream).await
        });
        tokio::task::yield_now().await;
        chunk_tx
            .send(Ok::<_, reqwest::Error>(Bytes::from_static(b"chunk")))
            .await
            .expect("chunk receiver should stay alive");
        let chunk = match reader.await.expect("reader task should join") {
            Ok(chunk) => chunk,
            Err(_) => panic!("uncancelled token should not stop chunk reads"),
        };

        assert_eq!(chunk, Bytes::from_static(b"chunk"));
    }

    #[tokio::test]
    async fn stats_update_send_wakes_on_cancellation_when_channel_is_full() {
        let (tx, _rx) = flume::bounded(1);
        tx.send(StatsAggregatorUpdate::EnableOpenAiFallback)
            .expect("seed update should fill channel");
        let stop = CancellationToken::new();
        let task_stop = stop.clone();

        let task = tokio::spawn(async move {
            send_stats_update(&tx, StatsAggregatorUpdate::EnableOpenAiFallback, &task_stop).await
        });
        tokio::task::yield_now().await;
        stop.cancel();

        let sent = tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("stats update send should wake on cancellation")
            .expect("stats update send should not panic");
        assert!(!sent);
    }

    #[test]
    fn empty_engine_stats_chunks_force_reconnect_without_spinning() {
        let config = EngineStatsStreamConfig::default();

        assert!(matches!(
            non_empty_engine_stats_chunk(&config, Bytes::new()),
            Err(StreamReadOutcome::Retry("empty_chunk"))
        ));
        assert_eq!(
            non_empty_engine_stats_chunk(&config, Bytes::from_static(b"chunk"))
                .expect("non-empty chunks should continue draining"),
            Bytes::from_static(b"chunk")
        );
    }

    #[tokio::test]
    async fn engine_stats_line_length_allows_exact_limit_only() {
        let line =
            br#"{"v":1,"type":"stats","request_id":"req-1","model":"model-a","tokens_generated":1}"#;
        let mut config = EngineStatsStreamConfig {
            max_line_bytes: line.len(),
            ..Default::default()
        };
        let stop = CancellationToken::new();
        let (tx, rx) = flume::bounded(1);
        let mut valid_event_seen = false;
        let mut line_buffer = line.to_vec();
        line_buffer.push(b'\n');

        assert!(
            process_buffered_engine_stats_lines(
                &config,
                &tx,
                &mut valid_event_seen,
                &mut line_buffer,
                &stop,
            )
            .await
            .is_none()
        );
        assert!(valid_event_seen);
        assert!(line_buffer.is_empty());
        assert!(matches!(
            rx.try_recv()
                .expect("exact-limit line should publish stats"),
            StatsAggregatorUpdate::RequestCounters(_)
        ));

        let (tx, rx) = flume::bounded(1);
        let mut valid_event_seen = false;
        let mut line_buffer = br#"{"v":1,"type":"ping"}"#.to_vec();
        line_buffer.push(b'\n');
        line_buffer.extend_from_slice(line);
        line_buffer.push(b'\n');

        assert!(
            process_buffered_engine_stats_lines(
                &config,
                &tx,
                &mut valid_event_seen,
                &mut line_buffer,
                &stop,
            )
            .await
            .is_none()
        );
        assert!(valid_event_seen);
        assert!(line_buffer.is_empty());
        assert!(matches!(
            rx.try_recv()
                .expect("exact-limit line should publish after a prior line"),
            StatsAggregatorUpdate::RequestCounters(_)
        ));

        config.max_line_bytes = line.len() - 1;
        let (tx, rx) = flume::bounded(1);
        let mut valid_event_seen = false;
        let mut line_buffer = line.to_vec();
        line_buffer.push(b'\n');

        assert!(
            process_buffered_engine_stats_lines(
                &config,
                &tx,
                &mut valid_event_seen,
                &mut line_buffer,
                &stop,
            )
            .await
            .is_none()
        );
        assert!(!valid_event_seen);
        assert!(line_buffer.is_empty());
        assert!(rx.try_recv().is_err());

        let metrics = PylonMetrics::new().expect("metrics should initialize");
        config.metrics = Some(metrics.clone());
        let (tx, rx) = flume::bounded(1);
        let mut valid_event_seen = false;
        let mut line_buffer = line.to_vec();
        line_buffer.push(b'\n');
        line_buffer.extend_from_slice(br#"{"v":1,"type":"ping"}"#);
        line_buffer.push(b'\n');

        assert!(
            process_buffered_engine_stats_lines(
                &config,
                &tx,
                &mut valid_event_seen,
                &mut line_buffer,
                &stop,
            )
            .await
            .is_none()
        );
        assert!(valid_event_seen);
        assert!(line_buffer.is_empty());
        assert!(rx.try_recv().is_err());
        let body = metrics.gather_text().expect("metrics should encode");
        assert!(body.contains(
            r#"pylon_engine_stats_stream_invalid_events_total{reason="line_too_large"} 1"#
        ));
        assert!(body.contains(r#"pylon_engine_stats_stream_events_total{type="ping"} 1"#));
        assert!(!body.contains(r#"pylon_engine_stats_stream_invalid_events_total{reason="json"}"#));
    }

    #[tokio::test]
    async fn blank_lf_and_crlf_engine_stats_lines_are_ignored() {
        let metrics = PylonMetrics::new().expect("metrics should initialize");
        let config = EngineStatsStreamConfig {
            metrics: Some(metrics.clone()),
            ..Default::default()
        };
        let stop = CancellationToken::new();
        let (tx, rx) = flume::bounded(1);
        let mut valid_event_seen = false;
        let mut line_buffer = b"\n\r\n".to_vec();
        line_buffer.extend_from_slice(
            br#"{"v":1,"type":"stats","request_id":"req-1","model":"model-a","tokens_generated":1}"#,
        );
        line_buffer.extend_from_slice(b"\r\n");

        assert!(
            process_buffered_engine_stats_lines(
                &config,
                &tx,
                &mut valid_event_seen,
                &mut line_buffer,
                &stop,
            )
            .await
            .is_none()
        );

        assert!(valid_event_seen);
        assert!(line_buffer.is_empty());
        assert!(matches!(
            rx.try_recv()
                .expect("stats line after blank CRLF should publish"),
            StatsAggregatorUpdate::RequestCounters(_)
        ));
        let body = metrics.gather_text().expect("metrics should encode");
        assert!(!body.contains(r#"pylon_engine_stats_stream_invalid_events_total{reason="json"}"#));
    }

    #[test]
    fn compact_engine_stats_line_buffer_keeps_partial_tail() {
        let config = EngineStatsStreamConfig::default();
        let mut line_buffer = b"{\"v\":1,\"type\":\"ping\"}\n{\"v\":1".to_vec();

        compact_engine_stats_line_buffer(&config, &mut line_buffer, 22);

        assert_eq!(line_buffer, br#"{"v":1"#);
        compact_engine_stats_line_buffer(&config, &mut line_buffer, 6);
        assert!(line_buffer.is_empty());

        let config = EngineStatsStreamConfig {
            max_line_bytes: 4,
            ..Default::default()
        };
        let mut line_buffer = b"1234".to_vec();
        compact_engine_stats_line_buffer(&config, &mut line_buffer, 0);
        assert_eq!(line_buffer, b"1234");

        let mut line_buffer = b"12345".to_vec();
        compact_engine_stats_line_buffer(&config, &mut line_buffer, 0);
        assert!(line_buffer.is_empty());
    }

    #[tokio::test]
    async fn auto_mode_enables_openai_fallback_when_endpoint_is_unsupported_before_events() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let addr = listener.local_addr().expect("listener should have addr");
        let server = tokio::spawn(async move {
            let app = Router::new().route(
                "/pylon/v1/stats/stream",
                get(|| async { StatusCode::NOT_FOUND }),
            );
            axum::serve(listener, app).await.expect("server should run");
        });

        let (tx, rx) = flume::bounded(1);
        let mut config = EngineStatsStreamConfig::new(
            &format!("http://{addr}"),
            "/pylon/v1/stats/stream",
            EngineStatsStreamMode::Auto,
        );
        config.initial_reconnect_backoff = Duration::from_millis(1);
        config.max_reconnect_backoff = Duration::from_millis(1);

        let handle = start_engine_stats_stream(config, tx).expect("auto stats stream should start");
        let update = tokio::time::timeout(Duration::from_secs(2), rx.recv_async())
            .await
            .expect("auto mode should enable fallback")
            .expect("control update should be sent");

        assert!(matches!(
            update,
            StatsAggregatorUpdate::EnableOpenAiFallback
        ));

        handle.shutdown().await;
        server.abort();
    }

    #[tokio::test]
    async fn required_mode_does_not_enable_openai_fallback_for_unsupported_endpoint() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let addr = listener.local_addr().expect("listener should have addr");
        let server = tokio::spawn(async move {
            let app = Router::new().route(
                "/pylon/v1/stats/stream",
                get(|| async { StatusCode::NOT_FOUND }),
            );
            axum::serve(listener, app).await.expect("server should run");
        });

        let (tx, rx) = flume::bounded(1);
        let mut config = EngineStatsStreamConfig::new(
            &format!("http://{addr}"),
            "/pylon/v1/stats/stream",
            EngineStatsStreamMode::Required,
        );
        config.initial_reconnect_backoff = Duration::from_millis(1);
        config.max_reconnect_backoff = Duration::from_millis(1);

        let handle =
            start_engine_stats_stream(config, tx).expect("required stats stream should start");
        assert!(
            tokio::time::timeout(Duration::from_millis(50), rx.recv_async())
                .await
                .is_err(),
            "required mode must not enable OpenAI fallback"
        );

        handle.shutdown().await;
        server.abort();
    }

    #[tokio::test]
    async fn auto_mode_retries_unsupported_endpoint_after_valid_event_without_fallback() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let addr = listener.local_addr().expect("listener should have addr");
        let attempts = Arc::new(AtomicUsize::new(0));
        let server_attempts = attempts.clone();
        let server = tokio::spawn(async move {
            let app = Router::new().route(
                "/pylon/v1/stats/stream",
                get(move || {
                    let attempts = server_attempts.clone();
                    async move {
                        if attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                            (
                                StatusCode::OK,
                                "{\"v\":1,\"type\":\"stats\",\"request_id\":\"req-1\",\"model\":\"model-a\",\"tokens_generated\":1}\n",
                            )
                        } else {
                            (StatusCode::NOT_FOUND, "")
                        }
                    }
                }),
            );
            axum::serve(listener, app).await.expect("server should run");
        });

        let (tx, rx) = flume::bounded(4);
        let mut config = EngineStatsStreamConfig::new(
            &format!("http://{addr}"),
            "/pylon/v1/stats/stream",
            EngineStatsStreamMode::Auto,
        );
        config.initial_reconnect_backoff = Duration::from_millis(1);
        config.max_reconnect_backoff = Duration::from_millis(1);

        let handle = start_engine_stats_stream(config, tx).expect("auto stats stream should start");
        let update = tokio::time::timeout(Duration::from_secs(2), rx.recv_async())
            .await
            .expect("valid stats event should be sent")
            .expect("stats update should be sent");
        assert!(matches!(update, StatsAggregatorUpdate::RequestCounters(_)));
        assert!(
            tokio::time::timeout(Duration::from_millis(50), rx.recv_async())
                .await
                .is_err(),
            "auto mode must not switch to fallback after any valid stream event"
        );

        handle.shutdown().await;
        server.abort();
    }

    #[tokio::test]
    async fn stream_invalid_events_record_metric_reasons_before_valid_update() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let addr = listener.local_addr().expect("listener should have addr");
        let server = tokio::spawn(async move {
            let app = Router::new().route(
                "/pylon/v1/stats/stream",
                get(|| async {
                    concat!(
                        "not-json\n",
                        "[1]\n",
                        "{\"type\":\"ping\"}\n",
                        "{\"v\":2,\"type\":\"ping\"}\n",
                        "{\"v\":1,\"type\":\"nope\"}\n",
                        "{\"v\":1,\"type\":\"stats\",\"request_id\":\"req-empty\",\"model\":\"model-a\"}\n",
                        "{\"v\":1,\"type\":\"stats\",\"request_id\":\"\",\"model\":\"model-a\",\"tokens_generated\":1}\n",
                        "{\"v\":1,\"type\":\"stats\",\"request_id\":\"req-valid\",\"model\":\"model-a\",\"tokens_generated\":1}\n",
                    )
                }),
            );
            axum::serve(listener, app).await.expect("server should run");
        });

        let metrics = PylonMetrics::new().expect("metrics should initialize");
        let (tx, rx) = flume::bounded(4);
        let mut config = EngineStatsStreamConfig::new(
            &format!("http://{addr}"),
            "/pylon/v1/stats/stream",
            EngineStatsStreamMode::Required,
        );
        config.initial_reconnect_backoff = Duration::from_secs(60);
        config.max_reconnect_backoff = Duration::from_secs(60);
        config.metrics = Some(metrics.clone());

        let handle =
            start_engine_stats_stream(config, tx).expect("required stats stream should start");
        let update = tokio::time::timeout(Duration::from_secs(2), rx.recv_async())
            .await
            .expect("valid stats event should be sent")
            .expect("stats update should be sent");
        let StatsAggregatorUpdate::RequestCounters(update) = update else {
            panic!("expected valid stream line to produce request counters");
        };
        assert_eq!(update.request_id, "req-valid");
        assert_eq!(update.tokens_generated, Some(1));

        handle.shutdown().await;
        server.abort();

        let body = metrics.gather_text().expect("metrics should encode");
        for reason in [
            "json",
            "not_object",
            "missing_field",
            "version",
            "type",
            "empty_stats",
            "field",
        ] {
            assert!(
                body.contains(&format!(
                    r#"pylon_engine_stats_stream_invalid_events_total{{reason="{reason}"}} 1"#
                )),
                "missing invalid-event metric for reason {reason}; body:\n{body}"
            );
        }
        assert!(body.contains(r#"pylon_engine_stats_stream_events_total{type="stats"} 1"#));
    }
}
