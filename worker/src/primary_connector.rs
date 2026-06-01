// Copyright(C) Facebook, Inc. and its affiliates.
use crate::worker::SerializedBatchDigestMessage;
use bytes::Bytes;
use log::debug;
use network::CancelHandler;
use network::{ReliableSender, SimpleSender};
use std::net::SocketAddr;
use tokio::sync::mpsc::Receiver;

// Send batches' digests to the primary.
pub struct PrimaryConnector {
    /// The primary network address.
    primary_address: SocketAddr,
    /// Input channel to receive the digests to send to the primary.
    rx_digest: Receiver<SerializedBatchDigestMessage>,
    /// A network sender to send the baches' digests to the primary.
    network: SimpleSender,
    //network: ReliableSender,
    // Cancel handlers
    cancel_handlers: Vec<CancelHandler>,
}

impl PrimaryConnector {
    pub fn spawn(primary_address: SocketAddr, rx_digest: Receiver<SerializedBatchDigestMessage>) {
        tokio::spawn(async move {
            Self {
                primary_address,
                rx_digest,
                network: SimpleSender::new(),
                //network: ReliableSender::new(),
                cancel_handlers: Vec::new(),
            }
            .run()
            .await;
        });
    }

    async fn run(&mut self) {
        while let Some(digest) = self.rx_digest.recv().await {
            // Send the digest through the network.
            debug!("Received primary connector batch digest {:?}", digest);
            /*let handler = self.network
                .send(self.primary_address, Bytes::from(digest))
                .await;
            self.cancel_handlers.push(handler);*/

            self.network
                .send(self.primary_address, Bytes::from(digest))
                .await;
        }
    }
}
