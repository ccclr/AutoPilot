// Copyright(C) Facebook, Inc. and its affiliates.
use anyhow::{Context, Result};
use bytes::BufMut as _;
use bytes::BytesMut;
use clap::{crate_name, crate_version, App, AppSettings};
use env_logger::Env;
use futures::future::join_all;
use futures::sink::SinkExt as _;
use log::{info, warn};
use rand::Rng;
use std::net::SocketAddr;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::net::TcpStream;
use tokio::time::{interval, sleep, Duration, Instant};
use tokio_util::codec::{Framed, LengthDelimitedCodec};

#[tokio::main]
async fn main() -> Result<()> {
    let matches = App::new(crate_name!())
        .version(crate_version!())
        .about("Benchmark client for Sailfish.")
        .args_from_usage("<ADDR> 'The network address of the node where to send txs'")
        .args_from_usage("--size=<INT> 'The size of each transaction in bytes'")
        .args_from_usage("--rate=<INT> 'The rate (txs/s) at which to send the transactions'")
        .args_from_usage("--nodes=[ADDR]... 'Network addresses that must be reachable before starting the benchmark.'")
        .args_from_usage("--node-id=<INT> 'The index of this client node (0-based)'")
        .args_from_usage("--hotspot-windows=[WINDOW]... 'Hotspot time windows in format start:end (e.g., 10:20)'")
        .args_from_usage("--hotspot-node-ids=[IDS] 'Resolved hotspot node ids per window, e.g. \"2,5|3,4\"'")
        .args_from_usage("--hotspot-node-rates=[RATES] 'Per-window per-node hotspot rates, e.g. \"0.9,0.7|0.5,0.4\"'")
        .setting(AppSettings::ArgRequiredElseHelp)
        .get_matches();

    env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    let target = matches
        .value_of("ADDR")
        .unwrap()
        .parse::<SocketAddr>()
        .context("Invalid socket address format")?;
    let size = matches
        .value_of("size")
        .unwrap()
        .parse::<usize>()
        .context("The size of transactions must be a non-negative integer")?;
    let rate = matches
        .value_of("rate")
        .unwrap()
        .parse::<u64>()
        .context("The rate of transactions must be a non-negative integer")?;
    let nodes = matches
        .values_of("nodes")
        .unwrap_or_default()
        .into_iter()
        .map(|x| x.parse::<SocketAddr>())
        .collect::<Result<Vec<_>, _>>()
        .context("Invalid socket address format")?;

    let node_id = matches
        .value_of("node-id")
        .unwrap_or("0")
        .parse::<usize>()
        .context("Node ID must be a non-negative integer")?;

    // Parse hotspot configuration
    let hotspot_config = parse_hotspot_config(&matches)?;

    info!("Node address: {}", target);
    info!("Node ID: {}", node_id);
    info!("Total nodes: {}", nodes.len());

    // NOTE: This log entry is used to compute performance.
    info!("Transactions size: {} B", size);

    // NOTE: This log entry is used to compute performance.
    info!("Transactions rate: {} tx/s", rate);

    if let Some(ref config) = hotspot_config {
        info!("Hotspot configuration enabled:");
        for (i, start_end) in config.hotspot_windows.iter().enumerate() {
            let (start, end) = *start_end;
            let num_info = match (&config.hotspot_node_ids, &config.hotspot_node_rates) {
                (Some(ids), Some(rates)) => format!("nodes {:?}, rates {:?}", ids[i], rates[i]),
                _ => "?".to_string(),
            };
            info!("  Window {}: {}s-{}s, {}", i + 1, start, end, num_info);
        }
    }

    let client = Client {
        target,
        size,
        rate,
        nodes,
        node_id,
        hotspot_config,
    };

    // Wait for all nodes to be online and synchronized.
    client.wait().await;

    // Start the benchmark.
    client.send().await.context("Failed to submit transactions")
}

fn parse_hotspot_config(matches: &clap::ArgMatches) -> Result<Option<HotspotConfig>> {
    let windows = matches.values_of("hotspot-windows");
    let node_ids_str = matches.value_of("hotspot-node-ids");
    let node_rates_str = matches.value_of("hotspot-node-rates");

    if windows.is_none() {
        return Ok(None);
    }

    let windows: Vec<(u64, u64)> = windows
        .unwrap()
        .map(|w| {
            let parts: Vec<&str> = w.split(':').collect();
            if parts.len() != 2 {
                return Err(anyhow::Error::msg("Invalid window format, use start:end"));
            }
            let start = parts[0].parse::<u64>().context("Invalid start time")?;
            let end = parts[1].parse::<u64>().context("Invalid end time")?;
            Ok((start, end))
        })
        .collect::<Result<_, _>>()?;

    let (hotspot_node_ids, hotspot_node_rates) = if let Some(ids_str) = node_ids_str {
        // Format "2,5|3,4" - pipe separates windows, comma within each window
        let per_window: Vec<Vec<usize>> = ids_str
            .split('|')
            .map(|s| {
                s.split(',')
                    .map(|n| {
                        n.trim()
                            .parse::<usize>()
                            .context("Invalid node id in hotspot-node-ids")
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        if per_window.len() != windows.len() {
            return Err(anyhow::Error::msg(
                "hotspot-node-ids must have same number of windows as hotspot-windows",
            ));
        }
        let per_window_rates: Vec<Vec<f64>> = if let Some(rates_str) = node_rates_str {
            let parsed: Vec<Vec<f64>> = rates_str
                .split('|')
                .map(|s| {
                    s.split(',')
                        .map(|r| r.trim().parse::<f64>().context("Invalid hotspot-node-rate"))
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?;
            if parsed.len() != windows.len() {
                return Err(anyhow::Error::msg(
                    "hotspot-node-rates must have same number of windows as hotspot-windows",
                ));
            }
            for (w, rates) in parsed.iter().enumerate() {
                if rates.len() != per_window[w].len() {
                    return Err(anyhow::Error::msg(
                        "hotspot-node-rates entries must match hotspot-node-ids entries per window",
                    ));
                }
            }
            parsed
        } else {
            return Err(anyhow::Error::msg(
                "hotspot-node-rates is required when hotspot-node-ids is provided",
            ));
        };
        (Some(per_window), Some(per_window_rates))
    } else {
        return Err(anyhow::Error::msg(
            "Hotspot config requires hotspot-node-ids and hotspot-node-rates",
        ));
    };

    Ok(Some(HotspotConfig {
        hotspot_windows: windows,
        hotspot_node_ids,
        hotspot_node_rates,
    }))
}

#[derive(Debug, Clone)]
pub struct HotspotConfig {
    pub hotspot_windows: Vec<(u64, u64)>, // [start, end] in seconds
    pub hotspot_node_ids: Option<Vec<Vec<usize>>>, // Resolved node ids per window
    pub hotspot_node_rates: Option<Vec<Vec<f64>>>, // Per-window per-node hotspot rate multipliers
}

impl HotspotConfig {
    /// Calculate the arrival rate for a given time and node, keeping the total rate constant
    pub fn get_arrival_rate(
        &self,
        elapsed_secs: u64,
        base_rate: f64,
        node_idx: usize,
        total_nodes: usize,
    ) -> f64 {
        let mut num_hotspot = 0;
        let mut rate_increase = 0.0;
        let mut is_hotspot = false;

        for (w, start_end) in self.hotspot_windows.iter().enumerate() {
            let (start, end) = *start_end;
            if elapsed_secs >= start && elapsed_secs <= end {
                if let Some(ref ids) = self.hotspot_node_ids {
                    let window_ids = &ids[w];
                    num_hotspot = window_ids.len();
                    if let Some(pos) = window_ids.iter().position(|&id| id == node_idx) {
                        is_hotspot = true;
                        if let Some(ref node_rates) = self.hotspot_node_rates {
                            if let Some(window_rates) = node_rates.get(w) {
                                if let Some(node_rate) = window_rates.get(pos) {
                                    rate_increase = *node_rate;
                                }
                            }
                        }
                    }
                }
                break;
            }
        }

        self.calculate_redistributed_rate(
            base_rate,
            total_nodes,
            num_hotspot,
            is_hotspot,
            rate_increase,
        )
    }

    fn calculate_redistributed_rate(
        &self,
        base_rate: f64,
        total_nodes: usize,
        num_hotspot: usize,
        is_hotspot: bool,
        rate_increase: f64,
    ) -> f64 {
        if num_hotspot >= total_nodes {
            return base_rate;
        }
        if !is_hotspot {
            base_rate
        } else {
            let mut normal_rate = base_rate * (1.0 - rate_increase);
            normal_rate
        }
    }
}

struct Client {
    target: SocketAddr,     // The network address of the node where to send txs
    size: usize,            // The size of each transaction in bytes
    rate: u64,              // The base sending rate
    nodes: Vec<SocketAddr>, // All node addresses
    node_id: usize,         // The index of this client node (0-based)
    hotspot_config: Option<HotspotConfig>, // Hotspot configuration
}

impl Client {
    pub async fn send(&self) -> Result<()> {
        const PRECISION: u64 = 20; // Sample precision.
        const BURST_DURATION: u64 = 1000 / PRECISION;

        // The transaction size must be at least 17 bytes to ensure all txs are different.
        // 1 byte (flag) + 8 bytes (counter) + 8 bytes (timestamp) = 17 bytes minimum
        if self.size < 17 {
            return Err(anyhow::Error::msg(
                "Transaction size must be at least 17 bytes to include timestamp",
            ));
        }

        // Connect to the mempool.
        let stream = TcpStream::connect(self.target)
            .await
            .context(format!("failed to connect to {}", self.target))?;

        let mut tx = BytesMut::with_capacity(self.size);
        let mut counter = 0;
        let mut transport = Framed::new(stream, LengthDelimitedCodec::new());
        let mut r = rand::thread_rng().gen();
        let start_time = Instant::now();
        let interval = interval(Duration::from_millis(BURST_DURATION));
        tokio::pin!(interval);

        // NOTE: This log entry is used to compute performance.
        info!("Start sending transactions");

        'main: loop {
            interval.as_mut().tick().await;
            let now = Instant::now();
            let elapsed_secs = start_time.elapsed().as_secs();

            let current_rate = if let Some(ref config) = self.hotspot_config {
                config.get_arrival_rate(
                    elapsed_secs,
                    self.rate as f64,
                    self.node_id,
                    self.nodes.len(),
                )
            } else {
                self.rate as f64
            };

            // Calculate the number of transactions to send in the current burst period
            let burst = (current_rate / PRECISION as f64).round() as u64;

            // Send transactions in the current burst period
            for x in 0..burst {
                // Get the current system timestamp (microseconds)
                let timestamp_us = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_micros() as u64;

                if x % 10000 == 0 {
                    // NOTE: This log entry is used to compute performance.
                    info!("Sending sample transaction {}", counter);

                    tx.put_u8(0u8); // Sample txs start with 0.
                    tx.put_u64(counter); // This counter identifies the tx.
                } else {
                    r += 1;
                    tx.put_u8(1u8); // Standard txs start with 1.
                    tx.put_u64(r); // Ensures all clients send different txs.
                };

                tx.resize(self.size, 0u8); //Truncate any bits past size
                let bytes = tx.split().freeze(); //split() moves byte content from tx to bytes (i.e. avoids copy). freeze() makes it const so it can be shared. (bytes can now be used/sent async)
                                                 //Note: Does not sign transactions. Transaction id-s are not unique w.r.t to content.
                if let Err(e) = transport.send(bytes).await {
                    //Uses TCP connection to send request to assigned worker. Note: Optimistically only sending to one worker.
                    warn!("Failed to send transaction: {}", e);
                    break 'main;
                }

                // tx.put_u8(0u8); // Sample txs start with 0.
                // tx.put_u64(counter); // This counter identifies the tx.
                // tx.put_u64(timestamp_us); // Add timestamp for latency measurement

                // // Include node_id to help with aggregated throughput calculation
                // tx.put_u32(self.node_id as u32);

                // tx.resize(self.size, 0u8); // Truncate any bits past size
                // let bytes = tx.split().freeze(); // split() moves byte content from tx to bytes

                // // Send transaction
                // if let Err(e) = transport.send(bytes).await {
                //     warn!("Failed to send transaction: {}", e);
                //     continue;
                // }

                counter += 1;
            }

            // Check if sending time is too long
            if now.elapsed().as_millis() > BURST_DURATION as u128 {
                // NOTE: This log entry is used to compute performance.
                warn!("Transaction rate too high for this client");
            }
        }
        Ok(())
    }

    pub async fn wait(&self) {
        // Wait for all nodes to be online.
        info!("Waiting for all nodes to be online...");
        join_all(self.nodes.iter().cloned().map(|address| {
            tokio::spawn(async move {
                while TcpStream::connect(address).await.is_err() {
                    sleep(Duration::from_millis(10)).await;
                }
            })
        }))
        .await;
    }
}
