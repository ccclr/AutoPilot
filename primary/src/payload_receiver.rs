// Copyright(C) Facebook, Inc. and its affiliates.
use config::WorkerId;
use crypto::Digest;
use log::debug;
use store::Store;
use tokio::sync::mpsc::Receiver;

/// Extracts node index from store path (e.g., ".db-3" -> 3)
fn extract_node_index_from_store_path(store_path: &str) -> Option<usize> {
    if let Some(start) = store_path.find(".db-") {
        let remaining = &store_path[start + 4..]; // Skip ".db-"
        if let Some(end) = remaining.find(|c: char| !c.is_ascii_digit()) {
            remaining[..end].parse::<usize>().ok()
        } else {
            remaining.parse::<usize>().ok()
        }
    } else {
        None
    }
}

/// Receives batches' digests of other authorities. These are only needed to verify incoming
/// headers (ie. make sure we have their payload).
pub struct PayloadReceiver {
    /// The persistent storage.
    store: Store,
    /// Receives batches' digests from the network.
    rx_workers: Receiver<(Digest, WorkerId, config::BatchMetadata)>,
    /// Node index for identification purposes.
    node_index: usize,
}

impl PayloadReceiver {
    pub fn spawn(
        store: Store,
        store_path: String,
        rx_workers: Receiver<(Digest, WorkerId, config::BatchMetadata)>,
    ) {
        let node_index = extract_node_index_from_store_path(&store_path).unwrap_or(0); // Default to 0 if parsing fails

        tokio::spawn(async move {
            Self {
                store,
                rx_workers,
                node_index,
            }
            .run()
            .await;
        });
    }

    async fn run(&mut self) {
        debug!(
            "PayloadReceiver initialized with node_index: {}",
            self.node_index
        );

        while let Some((digest, worker_id, _metadata)) = self.rx_workers.recv().await {
            debug!("Receive Digest: {} from node {}", digest, self.node_index);
            let key = [digest.as_ref(), &worker_id.to_le_bytes()].concat();
            self.store.write(key.to_vec(), Vec::default()).await;
            debug!("Wrote Digest: {}", digest);
        }
    }
}
