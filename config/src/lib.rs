// Copyright(C) Facebook, Inc. and its affiliates.
use crypto::{generate_production_keypair, Hash, PublicKey, SecretKey};
use log::info;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::{vec_deque, BTreeMap, HashMap, VecDeque};
use std::fs::{self, OpenOptions};
use std::io::BufWriter;
use std::io::Write as _;
use std::net::SocketAddr;
use std::sync::Mutex;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Node {0} is not in the committee")]
    NotInCommittee(PublicKey),

    #[error("Unknown worker id {0}")]
    UnknownWorker(WorkerId),

    #[error("Failed to read config file '{file}': {message}")]
    ImportError { file: String, message: String },

    #[error("Failed to write config file '{file}': {message}")]
    ExportError { file: String, message: String },
}

pub trait Import: DeserializeOwned {
    fn import(path: &str) -> Result<Self, ConfigError> {
        let reader = || -> Result<Self, std::io::Error> {
            let data = fs::read(path)?;
            Ok(serde_json::from_slice(data.as_slice())?)
        };
        reader().map_err(|e| ConfigError::ImportError {
            file: path.to_string(),
            message: e.to_string(),
        })
    }
}

pub trait Export: Serialize {
    fn export(&self, path: &str) -> Result<(), ConfigError> {
        let writer = || -> Result<(), std::io::Error> {
            let file = OpenOptions::new().create(true).write(true).open(path)?;
            let mut writer = BufWriter::new(file);
            let data = serde_json::to_string_pretty(self).unwrap();
            writer.write_all(data.as_ref())?;
            writer.write_all(b"\n")?;
            Ok(())
        };
        writer().map_err(|e| ConfigError::ExportError {
            file: path.to_string(),
            message: e.to_string(),
        })
    }
}

pub type Stake = u32;
pub type WorkerId = u32;

#[derive(Deserialize, Clone)]
pub struct Parameters {
    /// The timeout delay of the consensus protocol.
    pub timeout_delay: u64,
    /// The preferred header size. The primary creates a new header when it has enough parents and
    /// enough batches' digests to reach `header_size`. Denominated in bytes.
    pub header_size: usize,
    /// The maximum delay that the primary waits between generating two headers, even if the header
    /// did not reach `max_header_size`. Denominated in ms.
    pub max_header_delay: u64,
    /// The depth of the garbage collection (Denominated in number of rounds).
    pub gc_depth: u64,
    /// The delay after which the synchronizer retries to send sync requests. Denominated in ms.
    pub sync_retry_delay: u64,
    /// Determine with how many nodes to sync when re-trying to send sync-request. These nodes
    /// are picked at random from the committee.
    pub sync_retry_nodes: usize,
    /// The preferred batch size. The workers seal a batch of transactions when it reaches this size.
    /// Denominated in bytes.
    pub batch_size: usize,
    /// The delay after which the workers seal a batch of transactions, even if `max_batch_size`
    /// is not reached. Denominated in ms.
    pub max_batch_delay: u64,

    //Autobahn protocol config parameters
    pub use_optimistic_tips: bool, //default = true (TODO: implement non optimistic tip option)

    pub use_parallel_proposals: bool, //default = true (TODO: implement sequential slot option)
    pub k: u64,                       //Max open conensus instances at a time.

    pub use_fast_path: bool, //default = false
    pub fast_path_timeout: u64,

    pub use_ride_share: bool,
    pub car_timeout: u64,

    pub cut_condition_type: u8,

    //asynchrony simulation:
    // pub simulate_asynchrony: bool,
    // pub asynchrony_start: u64,
    // pub asynchrony_duration: u64,
    pub simulate_asynchrony: bool,          //Simulating an async event
    pub asynchrony_type: VecDeque<u8>, //Type of effects: 0 for delay full async duration, 1 for partition, 2 for  failure, 3 for egress delay. Will start #type many blips.
    pub asynchrony_start: VecDeque<u64>, //Start of async period   //offset from current time (in seconds) when to start next async effect
    pub asynchrony_duration: VecDeque<u64>, //Duration of async period (seconds)
    pub affected_nodes: VecDeque<u64>,   ////first k nodes experience specified async behavior
    #[serde(default)]
    pub asynchrony_node_ids_per_window: VecDeque<Vec<u64>>, // Optional explicit node ids for each async window.

    pub egress_penalty: u64, //ms of delay
    #[serde(default)]
    pub egress_penalty_per_node: VecDeque<u64>, // Optional per-node egress penalty override.
    pub use_fast_sync: bool,
    pub use_exponential_timeouts: bool,

    // Metrics collection parameters
    pub epoch_slots: u64, // Number of slots per epoch (h parameter) - DEPRECATED
    pub window_size: u64, // Size of time window within each epoch (j parameter) - DEPRECATED

    /// Slot in each epoch after which parameter updates are applied.
    /// 0 means apply immediately when signal is detected.
    #[serde(default)]
    pub applied_begin: u64,

    // ── Ablation: aggregation strategy ──────────────────────────────────────
    /// Strategy used to aggregate per-lane growth_rates across StateReports.
    /// Options: "normal" (default), "mean".
    /// Default preserves the original "normal" behaviour so existing deployments
    /// are unaffected when the key is absent from the config file.
    #[serde(default)]
    pub aggregation_strategy: AggregationStrategy,

    // ── Ablation: data-pollution simulation ─────────────────────────────────
    /// Node indices (0-based) that act as polluters.  Empty by default.
    #[serde(default)]
    pub data_pollution_node_ids: Vec<u64>,
    /// Probability [0.0, 1.0] that a polluter reports fake metrics instead of
    /// real ones.  0.0 (default) means no pollution even for listed nodes.
    #[serde(default)]
    pub data_pollution_prob: f64,
    /// How selected polluters alter their StateReport metrics.
    #[serde(default)]
    pub data_pollution_strategy: DataPollutionStrategy,
}

/// How to aggregate per-lane `growth_rates` values across all StateReports.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AggregationStrategy {
    /// Original behaviour (max for growth_rates, median for reward and fast_path_ratio).
    Normal,
    /// Arithmetic mean across all reporting nodes for all three metrics.
    Mean,
}

impl Default for AggregationStrategy {
    fn default() -> Self {
        Self::Normal
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DataPollutionStrategy {
    /// Existing behaviour: randomly scale key metrics up or down.
    RandomScale,
    /// Make the reported metrics converge to a narrow, plausible band.
    MeanEqualize,
}

impl Default for DataPollutionStrategy {
    fn default() -> Self {
        Self::RandomScale
    }
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            timeout_delay: 1_000,
            header_size: 1_000,
            max_header_delay: 100,
            gc_depth: 50,
            sync_retry_delay: 5_000,
            sync_retry_nodes: 3,
            batch_size: 500_000,
            max_batch_delay: 100,

            //Autobahn microbench configs
            use_optimistic_tips: true,
            use_parallel_proposals: true,
            k: 4,
            use_fast_path: true,
            fast_path_timeout: 500,
            use_ride_share: false,
            car_timeout: 2000,

            cut_condition_type: 2,
            //Async simulation:
            // simulate_asynchrony: false,
            // asynchrony_start: 20_000, //20 second in
            // asynchrony_duration: 10_000, //10 seconds

            //Async simulation:
            simulate_asynchrony: false,
            asynchrony_type: vec![0].into(),
            asynchrony_start: vec![20_000].into(), //20 second in
            asynchrony_duration: vec![10].into(),  //10 seconds
            affected_nodes: vec![0].into(),
            asynchrony_node_ids_per_window: VecDeque::new(),

            egress_penalty: 0,
            egress_penalty_per_node: VecDeque::new(),
            use_fast_sync: false,
            use_exponential_timeouts: false,
            epoch_slots: 20, // Default: 20 slots per epoch - DEPRECATED
            window_size: 5,  // Default: 5 slots window size - DEPRECATED
            applied_begin: 0,
            aggregation_strategy: AggregationStrategy::Normal,
            data_pollution_node_ids: Vec::new(),
            data_pollution_prob: 0.0,
            data_pollution_strategy: DataPollutionStrategy::RandomScale,
        }
    }
}

impl Import for Parameters {}

impl Parameters {
    pub fn log(&self) {
        // NOTE: These log entries are needed to compute performance.
        info!("Timeout delay set to {} ms", self.timeout_delay);
        info!("Header size set to {} B", self.header_size);
        info!("Max header delay set to {} ms", self.max_header_delay);
        info!("Garbage collection depth set to {} rounds", self.gc_depth);
        info!("Sync retry delay set to {} ms", self.sync_retry_delay);
        info!("Sync retry nodes set to {} nodes", self.sync_retry_nodes);
        info!("Batch size set to {} B", self.batch_size);
        info!("Max batch delay set to {} ms", self.max_batch_delay);

        info!(
            "Fast path enabled? {}. Fast timeout: {}",
            self.use_fast_path, self.fast_path_timeout
        );
        info!("Optimistic tips enabled? {}", self.use_optimistic_tips);
        info!(
            "Parallel Proposals enabled? {}. K: {}",
            self.use_parallel_proposals, self.k
        );
        info!(
            "Ride share enabled? {}. Car timeout: {}",
            self.use_ride_share, self.car_timeout
        );
        info!("Cut condition type: {}", self.cut_condition_type);
        info!("Applied begin slot: {}", self.applied_begin);
        info!("Aggregation strategy: {:?}", self.aggregation_strategy);
        info!(
            "Data pollution: nodes={:?}, prob={}, strategy={:?}",
            self.data_pollution_node_ids, self.data_pollution_prob, self.data_pollution_strategy
        );
    }
}

#[derive(Clone, Deserialize)]
pub struct ConsensusAddresses {
    /// Address to receive messages from other consensus nodes (WAN).
    pub consensus_to_consensus: SocketAddr,
}

#[derive(Clone, Deserialize)]
pub struct PrimaryAddresses {
    /// Address to receive messages from other primaries (WAN).
    pub primary_to_primary: SocketAddr,
    /// Address to receive messages from our workers (LAN).
    pub worker_to_primary: SocketAddr,
}

#[derive(Clone, Deserialize, Eq, Hash, PartialEq)]
pub struct WorkerAddresses {
    /// Address to receive client transactions (WAN).
    pub transactions: SocketAddr,
    /// Address to receive messages from other workers (WAN).
    pub worker_to_worker: SocketAddr,
    /// Address to receive messages from our primary (LAN).
    pub primary_to_worker: SocketAddr,
}

#[derive(Clone, Deserialize)]
pub struct Authority {
    /// The voting power of this authority.
    pub stake: Stake,
    /// The network addresses of the consensus protocol.
    pub consensus: ConsensusAddresses,
    /// The network addresses of the primary.
    pub primary: PrimaryAddresses,
    /// Map of workers' id and their network addresses.
    pub workers: HashMap<WorkerId, WorkerAddresses>,
}

#[derive(Clone, Deserialize)]
pub struct Committee {
    pub authorities: BTreeMap<PublicKey, Authority>,
    pub id_map: HashMap<PublicKey, u64>, // Maps public key to node index
}

// Temporary struct for deserializing committee JSON without id_map
#[derive(Deserialize)]
struct CommitteeWithoutIdMap {
    pub authorities: BTreeMap<PublicKey, Authority>,
}

impl Import for Committee {
    fn import(path: &str) -> Result<Self, ConfigError> {
        let reader = || -> Result<Self, std::io::Error> {
            let data = fs::read(path)?;

            // First try to deserialize with id_map (for backward compatibility)
            let committee_result: Result<Self, _> = serde_json::from_slice(data.as_slice());

            let mut committee = match committee_result {
                Ok(c) => c,
                Err(_) => {
                    // If that fails, try deserializing without id_map and generate it
                    let temp_committee: CommitteeWithoutIdMap =
                        serde_json::from_slice(data.as_slice())?;

                    // Generate id_map based on the order of authorities in the JSON
                    let id_map = temp_committee
                        .authorities
                        .keys()
                        .enumerate()
                        .map(|(i, pk)| (*pk, i as u64))
                        .collect();

                    Self {
                        authorities: temp_committee.authorities,
                        id_map,
                    }
                }
            };

            // Always regenerate id_map to ensure it's correct
            committee.id_map = committee
                .authorities
                .keys()
                .enumerate()
                .map(|(i, pk)| (*pk, i as u64))
                .collect();

            Ok(committee)
        };
        reader().map_err(|e| ConfigError::ImportError {
            file: path.to_string(),
            message: e.to_string(),
        })
    }
}

impl Committee {
    pub fn new(info: Vec<(PublicKey, Stake, SocketAddr)>) -> Self {
        let mut id_map = HashMap::new();
        let authorities = info
            .into_iter()
            .enumerate()
            .map(|(i, (name, stake, address))| {
                id_map.insert(name, i as u64);
                let authority = Authority {
                    stake,
                    consensus: ConsensusAddresses {
                        consensus_to_consensus: address,
                    },
                    primary: PrimaryAddresses {
                        primary_to_primary: address,
                        worker_to_primary: address,
                    },
                    workers: HashMap::new(),
                };
                (name, authority)
            })
            .collect();

        Self {
            authorities,
            id_map,
        }
    }

    /// Returns the number of authorities.
    pub fn size(&self) -> usize {
        self.authorities.len()
    }

    /// Return the stake of a specific authority.
    pub fn stake(&self, name: &PublicKey) -> Stake {
        self.authorities.get(&name).map_or_else(|| 0, |x| x.stake)
    }

    /// Returns the stake of all authorities except `myself`.
    pub fn others_stake(&self, myself: &PublicKey) -> Vec<(PublicKey, Stake)> {
        self.authorities
            .iter()
            .filter(|(name, _)| name != &myself)
            .map(|(name, authority)| (*name, authority.stake))
            .collect()
    }

    /// Returns the stake required to reach a quorum (2f+1).
    pub fn quorum_threshold(&self) -> Stake {
        // If N = 3f + 1 + k (0 <= k < 3)
        // then (2 N + 3) / 3 = 2f + 1 + (2k + 2)/3 = 2f + 1 + k = N - f
        let total_votes: Stake = self.authorities.values().map(|x| x.stake).sum();
        2 * total_votes / 3 + 1
    }

    /// Returns the stake required to reach availability (f+1).
    pub fn validity_threshold(&self) -> Stake {
        // If N = 3f + 1 + k (0 <= k < 3)
        // then (N + 2) / 3 = f + 1 + k/3 = f + 1
        let total_votes: Stake = self.authorities.values().map(|x| x.stake).sum();
        (total_votes + 2) / 3
    }

    /// Returns the node index for a given public key.
    pub fn node_index(&self, name: &PublicKey) -> Option<u64> {
        self.id_map.get(name).copied()
    }

    pub fn fast_threshold(&self) -> Stake {
        let total_votes: Stake = self.authorities.values().map(|x| x.stake).sum();
        total_votes
    }

    /// Returns the consensus addresses of the target consensus node.
    pub fn consensus(&self, to: &PublicKey) -> Result<ConsensusAddresses, ConfigError> {
        self.authorities
            .get(to)
            .map(|x| x.consensus.clone())
            .ok_or_else(|| ConfigError::NotInCommittee(*to))
    }

    /// Returns the addresses of all consensus nodes except `myself`.
    pub fn others_consensus(&self, myself: &PublicKey) -> Vec<(PublicKey, ConsensusAddresses)> {
        self.authorities
            .iter()
            .filter(|(name, _)| name != &myself)
            .map(|(name, authority)| (*name, authority.consensus.clone()))
            .collect()
    }

    /// Returns the primary addresses of the target primary.
    pub fn primary(&self, to: &PublicKey) -> Result<PrimaryAddresses, ConfigError> {
        self.authorities
            .get(to)
            .map(|x| x.primary.clone())
            .ok_or_else(|| ConfigError::NotInCommittee(*to))
    }

    /// Returns the addresses of all primaries except `myself`.
    pub fn others_primaries(&self, myself: &PublicKey) -> Vec<(PublicKey, PrimaryAddresses)> {
        self.authorities
            .iter()
            .filter(|(name, _)| name != &myself)
            .map(|(name, authority)| (*name, authority.primary.clone()))
            .collect()
    }

    /// Returns the addresses of a specific worker (`id`) of a specific authority (`to`).
    pub fn worker(&self, to: &PublicKey, id: &WorkerId) -> Result<WorkerAddresses, ConfigError> {
        self.authorities
            .iter()
            .find(|(name, _)| name == &to)
            .map(|(_, authority)| authority)
            .ok_or_else(|| ConfigError::NotInCommittee(*to))?
            .workers
            .iter()
            .find(|(worker_id, _)| worker_id == &id)
            .map(|(_, worker)| worker.clone())
            .ok_or_else(|| ConfigError::NotInCommittee(*to))
    }

    /// Returns the addresses of all our workers.
    pub fn our_workers(&self, myself: &PublicKey) -> Result<Vec<WorkerAddresses>, ConfigError> {
        self.authorities
            .iter()
            .find(|(name, _)| name == &myself)
            .map(|(_, authority)| authority)
            .ok_or_else(|| ConfigError::NotInCommittee(*myself))?
            .workers
            .values()
            .cloned()
            .map(Ok)
            .collect()
    }

    /// Returns the addresses of all workers with a specific id except the ones of the authority
    /// specified by `myself`.
    pub fn others_workers(
        &self,
        myself: &PublicKey,
        id: &WorkerId,
    ) -> Vec<(PublicKey, WorkerAddresses)> {
        self.authorities
            .iter()
            .filter(|(name, _)| name != &myself)
            .filter_map(|(name, authority)| {
                authority
                    .workers
                    .iter()
                    .find(|(worker_id, _)| worker_id == &id)
                    .map(|(_, addresses)| (*name, addresses.clone()))
            })
            .collect()
    }

    pub fn address(&self, name: &PublicKey) -> Option<SocketAddr> {
        self.authorities
            .get(name)
            .map(|x| x.consensus.consensus_to_consensus)
    }

    pub fn broadcast_addresses(&self, myself: &PublicKey) -> Vec<(PublicKey, SocketAddr)> {
        self.authorities
            .iter()
            .filter(|(name, _)| name != &myself)
            .map(|(name, x)| (*name, x.consensus.consensus_to_consensus))
            .collect()
    }
}

#[derive(Serialize, Deserialize)]
pub struct KeyPair {
    /// The node's public key (and identifier).
    pub name: PublicKey,
    /// The node's secret key.
    pub secret: SecretKey,
}

impl Import for KeyPair {}
impl Export for KeyPair {}

impl KeyPair {
    pub fn new() -> Self {
        let (name, secret) = generate_production_keypair();
        Self { name, secret }
    }
}

impl Default for KeyPair {
    fn default() -> Self {
        Self::new()
    }
}

/// The message type received by clients
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Transaction {
    pub data: Vec<u8>,
    pub created_at: u128, // timestamp in microseconds since epoch
    pub size: usize,
}

/// The message type forwarded to quorum waiters
pub type Batch = Vec<Transaction>;

/// Metadata for a batch that is included in consensus headers for performance analysis
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchMetadata {
    pub author: PublicKey,               // the author/origin of this batch
    pub sample_tx_ids: Vec<u64>,         // IDs of sample transactions in this batch
    pub sample_tx_timestamps: Vec<u128>, // send_timestamps of sample transactions
    pub sample_tx_sizes: Vec<usize>,     // sizes of sample transactions in bytes
    pub transaction_count: usize,        // number of transactions in the batch
    pub batch_size: usize,               // total size of the batch in bytes
    pub avg_transaction_size: usize,     // average size of transactions in this batch
}

/// Metrics event for dedicated logging to reduce I/O overhead
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetricsEvent {
    pub timestamp: String,  // ISO format
    pub event_type: String, // 'batch', 'commit', 'transaction', etc.
    pub details: serde_json::Value,
}

/// Dedicated logger for metrics events
pub struct MetricsLogger {
    log_dir: std::path::PathBuf,
    metrics_log_path: std::path::PathBuf,
    buffer: Mutex<Vec<String>>,
    buffer_size: usize,
}

/// Helper function to generate metrics log filename based on node index
fn generate_metrics_filename(node_index: Option<usize>) -> String {
    match node_index {
        Some(idx) => format!("metrics-{}.log", idx),
        None => "metrics.log".to_string(),
    }
}

impl MetricsLogger {
    pub fn new(log_dir: &str) -> Self {
        Self::new_with_node_index(log_dir, None)
    }

    pub fn new_with_node_index(log_dir: &str, node_index: Option<usize>) -> Self {
        let log_dir = std::path::PathBuf::from(log_dir);
        let metrics_log_path = log_dir.join(generate_metrics_filename(node_index));

        // Ensure log directory exists
        if let Err(e) = std::fs::create_dir_all(&log_dir) {
            log::warn!("Failed to create metrics log directory: {}", e);
        }

        Self {
            log_dir,
            metrics_log_path,
            buffer: Mutex::new(Vec::new()),
            buffer_size: 1, // Flush immediately after each event
        }
    }

    pub fn log_event(&self, event_type: &str, details: serde_json::Value) {
        let timestamp = chrono::Utc::now().to_rfc3339();

        let event = MetricsEvent {
            timestamp,
            event_type: event_type.to_string(),
            details,
        };

        // Convert to JSON line
        match serde_json::to_string(&event) {
            Ok(json_line) => {
                let mut buffer = self.buffer.lock().unwrap();
                buffer.push(json_line);

                // Flush if buffer is full
                if buffer.len() >= self.buffer_size {
                    self.flush_buffer(&mut buffer);
                }
            }
            Err(e) => {
                log::warn!("Failed to serialize metrics event: {}", e);
            }
        }
    }

    fn flush_buffer(&self, buffer: &mut Vec<String>) {
        if buffer.is_empty() {
            return;
        }

        let content = buffer
            .iter()
            .map(|line| format!("{}\n", line))
            .collect::<String>();

        match OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.metrics_log_path)
        {
            Ok(mut file) => {
                if let Err(e) = file.write_all(content.as_bytes()) {
                    log::warn!("Failed to write to metrics log: {}", e);
                }
            }
            Err(e) => {
                log::warn!("Failed to open metrics log file: {}", e);
            }
        }

        buffer.clear();
    }

    pub fn flush(&self) {
        let mut buffer = self.buffer.lock().unwrap();
        self.flush_buffer(&mut buffer);
    }
}

impl Drop for MetricsLogger {
    fn drop(&mut self) {
        self.flush();
    }
}

/// Global metrics logger instance - now allows updates
static METRICS_LOGGER: std::sync::OnceLock<Mutex<Option<MetricsLogger>>> =
    std::sync::OnceLock::new();

/// Global parameter update signal file path for core to notify workers
pub static PARAMETER_UPDATE_SIGNAL_FILE: &str = "/tmp/autobahn_param_update.signal";

/// Signal file path for RL agent to notify core about parameter updates
pub static RL_PARAMETER_UPDATE_SIGNAL_FILE: &str = "/tmp/autobahn_rl_param_update";

pub fn init_metrics_logger(log_dir: &str) {
    init_metrics_logger_with_node_index(log_dir, None);
}

pub fn init_metrics_logger_with_node_index(log_dir: &str, node_index: Option<usize>) {
    let logger = MetricsLogger::new_with_node_index(log_dir, node_index);

    // Initialize or update the global logger
    let mutex = METRICS_LOGGER.get_or_init(|| Mutex::new(None));
    *mutex.lock().unwrap() = Some(logger);

    let log_filename = generate_metrics_filename(node_index);
    log::info!(
        "Initialized metrics logger (node_index: {:?}, log file: {})",
        node_index,
        log_filename
    );
}

pub fn get_metrics_logger() -> Option<std::sync::MutexGuard<'static, Option<MetricsLogger>>> {
    METRICS_LOGGER.get().map(|mutex| mutex.lock().unwrap())
}
