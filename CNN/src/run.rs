extern crate tch;

use std::vec;

use crate::io::*;
use crate::network::Network;
use crate::run::tch::nn::ModuleT;
use crate::run::tch::nn::OptimizerConfig;
use indicatif::{ProgressBar, ProgressStyle};
use plotters::prelude::*;
use tch::nn;
use tch::Device;

pub fn run() {
    let arguments: io::Argument = io::read_configs("Config.toml").unwrap();

    //check if cuda is available
    let device = Device::cuda_if_available();

    println!(
        "\n Device: {:?} \n Log interval: {:?} \n Epochs: {:?} \n Batch size: {:?}",
        device, arguments.log_interval, arguments.epochs, arguments.batch_size
    );
    // create a new optimizer
    let vs = nn::VarStore::new(device);
    // create a new network
    let net = Network::new(&vs.root());
    let mut net_opt: nn::Optimizer = nn::Adam::default().build(&vs, arguments.lr).unwrap();
    // create a new dataset
    let m: tch::vision::dataset::Dataset = tch::vision::mnist::load_dir("data").unwrap();

    // train the network
    let mut breaked = false;
    let pb = ProgressBar::new(arguments.epochs as u64);
    // change the pb style
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.green/green} {pos:>7}/{len:7} {msg}")
            .unwrap()
            .progress_chars("██▓"),
    );

    // create a vector to store the accuracy and time
    let mut accuracy_and_time: Vec<(f32, u128)> = vec![];
    accuracy_and_time.push((0.0, 0));

    let start = std::time::Instant::now();

    for _ in 0..arguments.epochs {
        if breaked {
            break;
        }
        let mut binding = m.train_iter(arguments.batch_size);
        let train_iter = binding.shuffle().to_device(device);
        let mut iteration_count = 0;

        let style = ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.yellow/yellow} {percent}% {msg}")
            .unwrap()
            .progress_chars("##-");
        let pb2 = ProgressBar::new(60_000 / arguments.batch_size as u64).with_style(style);

        for (bimages, blabels) in &mut *train_iter {
            pb2.inc(1);
            let bimages = bimages.to_device(device);
            let blabels = blabels.to_device(device);
            let output = net.forward_t(&bimages, true);
            let loss = output.cross_entropy_for_logits(&blabels);
            net_opt.backward_step(&loss);

            if iteration_count % arguments.log_interval == 0 {
                let accuracy = output.accuracy_for_logits(&blabels);
                // convert the accuracy tensor to a float value
                let accuracy_value = accuracy.double_value(&[]);
                accuracy_and_time.push((accuracy_value as f32, start.elapsed().as_millis()));
                pb2.set_message(
                    format!(
                        "| Iteration: {:?} | Loss: {:?} | Accuracy: {}",
                        iteration_count, loss, accuracy_value
                    )
                    .clone(),
                );
                if accuracy_value > 0.99 {
                    println!("Training accuracy reached over 99%! Breaking...");
                    breaked = true;
                    break;
                }
            }
            iteration_count += 1;
        }
        pb2.finish();
        pb.inc(1);
    }
    pb.finish();

    // test the network
    let test_accuracy: f64 = net.batch_accuracy_for_logits(
        &m.test_images,
        &m.test_labels,
        device,
        arguments.test_batch_size,
    );
    println!("Test Accuracy: {:?}", test_accuracy);
    draw_graph(accuracy_and_time.clone())
        .unwrap_or_else(|err| print!("Unable to draw graph: {:?}", err));
    println!("\n Done! \n Accuracy and time graph saved to accuracy.png");
}

fn draw_graph(accuracy_and_time: Vec<(f32, u128)>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\nDrawing graph...");
    let root = BitMapBackend::new("accuracy.png", (1920, 1080)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Accuracy over Time (ms)",
            ("sans-serif", 50).into_font().color(&BLACK),
        )
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0u64..accuracy_and_time.last().unwrap().1 as u64, 0f32..1f32)?;

    // Draw the accuracy line
    chart.draw_series(LineSeries::new(
        accuracy_and_time
            .iter()
            .map(|(x, y)| (*y as u64, *x))
            .collect::<Vec<_>>(),
        &BLACK,
    ))?;

    chart
        .configure_mesh()
        // We can customize the maximum number of labels allowed for each axis
        .x_labels(accuracy_and_time.len())
        .y_labels(10)
        // We can also change the format of the label text
        .y_label_formatter(&|x| format!("{:.1}", x))
        .draw()?;

    Ok(())
}

