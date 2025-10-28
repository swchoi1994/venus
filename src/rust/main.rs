use clap::Parser;
use std::path::PathBuf;
use tracing::info;
use venus_engine::{ApiServer, ServerConfig};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Directory containing model files
    #[arg(short, long, default_value = "./models")]
    model_dir: PathBuf,

    /// Host to bind to
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to bind to
    #[arg(short, long, default_value_t = 8000)]
    port: u16,

    /// Number of worker threads
    #[arg(short, long)]
    workers: Option<usize>,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Initialize logging
    let log_level = if args.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt().with_env_filter(log_level).init();

    info!("Starting Venus Inference Engine API Server");

    // Print platform information
    print_platform_info();

    // Create server configuration
    let config = ServerConfig {
        host: args.host,
        port: args.port,
        model_dir: args.model_dir,
        num_workers: args.workers.unwrap_or_else(num_cpus::get),
    };

    // Create and run server
    let server = ApiServer::new(config).await?;

    info!(
        "Server starting on http://{}:{}",
        server.config.host, server.config.port
    );

    server.run().await?;

    Ok(())
}

fn print_platform_info() {
    use venus_engine::{detect_platform, get_simd_features};

    info!("Platform: {:?}", detect_platform());
    info!("CPU Cores: {}", num_cpus::get());
    info!("SIMD Features: {:?}", get_simd_features());

    if let Ok(mem_info) = sys_info::mem_info() {
        info!("Total Memory: {} GB", mem_info.total / 1024 / 1024);
        info!("Available Memory: {} GB", mem_info.avail / 1024 / 1024);
    }
}
